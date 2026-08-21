"""Provider for Napisy24.

Napisy24 has two collections:

- *The catalogue* - a moderated collection.
  Each catalogue subtitle has an id, and the id maps to a subtitle page on the napisy24.pl website.
- *The program pool* - an unmoderated collection that users of the Napisy24 desktop program fill by uploading subtitles.
  A pool subtitle has no id (id=0), maps to no page on the website, and can be in any subtitle format.

Napisy24 has two APIs that share no convention, and they do not search the same collections:

- *The hash search* searches **both collections**.
  It takes the hash, the size and the name of the video file, and returns one subtitle or nothing.
  It sends the metadata and the subtitle bytes in one response, so it needs no download step.
  This provider sends ``n24pref=1``, which asks the service to prefer the catalogue copy when both hold one.
- *The catalogue search* searches **the catalogue only**.
  It takes an IMDB id or a title, needs no video file, and returns at most 25 records.
  A record holds metadata only, so the subtitle requires another call to the download endpoint.

Both collections are used, because the program pool holds subtitles that the catalogue does not.
The hash search runs first.
The catalogue search is the fallback, for a video with no hash, or when the hash search finds nothing.

"""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from http import HTTPStatus
from typing import TYPE_CHECKING, ClassVar
from zipfile import BadZipFile, ZipFile

from babelfish import Language  # type: ignore[import-untyped]
from requests import Session

from subliminal.exceptions import (
    AuthenticationError,
    NotInitializedProviderError,
    ProviderError,
    ServiceUnavailable,
)
from subliminal.matches import guess_matches
from subliminal.subtitle import SUBTITLE_EXTENSIONS, Subtitle
from subliminal.utils import decorate_imdb_id, ensure_list, safely_guessit, sanitize_id
from subliminal.video import Episode

from . import ParserBeautifulSoup, Provider

if TYPE_CHECKING:
    from collections.abc import Set
    from typing import Any, Protocol

    from bs4 import Tag
    from requests import Response

    from subliminal.video import Video

    class SubtitleMatcher(Protocol):
        """Matches the data of a subtitle against the data of a video.

        A match names one thing the two agree on, and subliminal scores it.
        """

        def get_matches(self, video: Video) -> set[str]:
            """Get the matches against the `video`."""
            ...


logger = logging.getLogger(__name__)

#: Name of the provider, and the key of its hash in ``video.hashes``
PROVIDER_NAME = 'napisy24'


# --------------------------------------------------------------------------------------------------
# Wire format
#
# The hash search sends a header of ``name:value`` fields and then a ZIP archive.
# The catalogue search sends XML fragments.
# Each type here turns one of the two into a typed record, so no text from the service is read elsewhere.
# --------------------------------------------------------------------------------------------------


class HashStatus(str, Enum):
    """The status at the start of a hash search response."""

    #: The service knows nothing about the video
    NOTHING_KNOWN = 'OK-0'

    #: The service knows the movie, but has no subtitle for it
    MOVIE_ONLY = 'OK-1'

    #: A subtitle follows the header
    FOUND = 'OK-2'

    #: A subtitle exists, but the ``tb`` parameter stopped the service from sending it.
    #: This provider never sends ``tb``, so the service never returns this status.
    NOT_SENT = 'OK-3'

    #: The account name or the password is wrong.
    #: The status is the whole body, and no field follows it.
    LOGIN_ERROR = 'login error'


@dataclass(frozen=True)
class HashResponse:
    """The response of the hash search: the metadata and the subtitle itself.

    Both arrive together, so the hash search needs no download step.

    Napisy24 uses its own format. Example:

    ``OK-2|res:1280x536|fps:23.976|napisId:71928|tlInfo:||PK<zip bytes>``

    Header fields are pipe-separated.
    The first field of the header is the status, and every other field is ``name:value``.
    A value may contain a colon, for example ``ftitle:brak imdb :-(``.
    Only the first colon separates the name from the value.

    The ``||`` sequence separates the header from the ZIP archive that holds the subtitle.
    """

    #: Separates the header from the archive
    RESPONSE_SEPARATOR: ClassVar[bytes] = b'||'

    #: Separates the fields inside the header
    FIELD_SEPARATOR: ClassVar[str] = '|'

    #: Separates the name of a field from its value
    NAME_SEPARATOR: ClassVar[str] = ':'

    #: What the service writes for an IMDB id that it does not know
    UNKNOWN_IMDB_IDS: ClassVar[tuple[str, ...]] = ('', '0')

    #: Id of the subtitle in the catalogue, or zero for a pool subtitle
    catalogue_id: int

    #: IMDB id of the movie, or None when the service does not know it
    imdb_id: str | None

    #: Frame rate of the video that the uploader had, which is not always the frame rate of the video to match
    frame_rate: float | None

    #: ZIP archive that holds the subtitle
    archive: bytes

    @classmethod
    def from_response(cls, content: bytes) -> HashResponse | None:
        """Read the response of the hash search.

        :param bytes content: the bytes that the service sent.
        :return: the subtitle and its data, or None when the service has no subtitle for the hash.
        :rtype: HashResponse | None
        :raise: :class:`~subliminal.exceptions.AuthenticationError` if the service refuses the account,
            :class:`~subliminal.exceptions.ProviderError` if the status is unknown or the header cannot be read.

        """
        header, _, archive = content.partition(cls.RESPONSE_SEPARATOR)
        text, *pairs = header.decode('utf-8', errors='replace').split(cls.FIELD_SEPARATOR)

        try:
            status = HashStatus(text)
        except ValueError as error:
            msg = f'Unknown status in the response of the hash search: {text!r}'
            raise ProviderError(msg) from error

        if status is HashStatus.LOGIN_ERROR:
            msg = 'The service refused the API account'
            raise AuthenticationError(msg)

        if status is not HashStatus.FOUND:
            logger.debug('The hash search found nothing, status %s', status.value)
            return None

        if not archive:
            msg = 'The hash search announced a subtitle but sent no archive'
            raise ProviderError(msg)

        try:
            fields = dict(pair.split(cls.NAME_SEPARATOR, 1) for pair in pairs)  # type: ignore[misc]
            imdb_id = fields['fimdb']
            frame_rate = fields['fps']

            return cls(
                catalogue_id=int(fields['napisId']),
                imdb_id=None if imdb_id in cls.UNKNOWN_IMDB_IDS else decorate_imdb_id(imdb_id),
                frame_rate=float(frame_rate) if frame_rate else None,
                archive=archive,
            )
        except (KeyError, ValueError) as error:
            msg = 'Cannot read the header of the hash search'
            raise ProviderError(msg) from error


def read_archive(archive: bytes) -> bytes | None:
    """Extract the subtitle from a ZIP archive that the service sent.

    An archive holds one subtitle, and next to it there can be a ``Napisy24.pl.url`` shortcut.
    The first member is then not the subtitle, so the member is chosen by extension.

    A catalogue entry for a range of episodes is the exception: its archive holds one subtitle for each episode.
    Only the first subtitle is read, so such an entry serves the first episode of the range and no other.

    :param bytes archive: the ZIP archive.
    :return: the content of the first subtitle, or None when no member is a subtitle.
    :rtype: bytes | None
    :raise: :class:`~subliminal.exceptions.ProviderError` if the bytes are not a ZIP archive.

    """
    try:
        with ZipFile(io.BytesIO(archive)) as zip_file:
            names = [name for name in zip_file.namelist() if name.lower().endswith(SUBTITLE_EXTENSIONS)]
            if not names:
                logger.warning('No subtitle in the archive, it holds these files: %r', zip_file.namelist())
                return None

            # TODO: read every subtitle of a range of episodes, and give each one the episode it belongs to
            return zip_file.read(names[0])

    except BadZipFile as error:
        msg = 'The service sent bytes that are not a ZIP archive'
        raise ProviderError(msg) from error


@dataclass(frozen=True)
class RecordElement:
    """One ``<subtitle>`` element of a catalogue response.

    A method returns None when the child element is absent, and also when the child element holds no text.
    """

    tag: Tag

    def text(self, name: str) -> str | None:
        """Read the text of a child element."""
        child = self.tag.find(name)
        value = child.get_text(strip=True) if child is not None else None
        return value or None

    def integer(self, name: str) -> int | None:
        """Read an integer from a child element."""
        value = self.text(name)
        return int(value) if value is not None else None

    def number(self, name: str) -> float | None:
        """Read a floating point number from a child element.

        The catalogue can write a comma as the decimal separator, as in ``23,976``.
        """
        value = self.text(name)
        return float(value.replace(',', '.')) if value is not None else None


@dataclass(frozen=True)
class TitleMetadata:
    """The metadata that the catalogue packs into one title element.

    The title of an episode contains a season and an episode suffix, for example ``Game of Thrones 3x10``.
    A movie title carries a name only.
    """

    #: Matches the suffix that the catalogue appends
    EPISODE_TOKEN: ClassVar[re.Pattern] = re.compile(r'\s\d{1,2}x\d{1,3}\b')

    #: The title, with the season and the episode removed
    title: str | None

    #: Season number, or None when the title carries none
    season: int | None

    #: Episode number, or None when the title carries none
    episode: int | None

    @classmethod
    def from_title(cls, title: str | None) -> TitleMetadata:
        """Split a title into its parts."""
        # guessit forced to ``episode`` reads any trailing number as a season and an episode.
        # It runs only when the suffix is there, so a movie title is left alone.
        if title is None or not cls.EPISODE_TOKEN.search(title):
            return cls(title=title, season=None, episode=None)

        guess = safely_guessit(title, {'type': 'episode'})
        return cls(
            title=guess.get('title') or title,
            season=guess.get('season'),
            episode=min(ensure_list(guess.get('episode')), default=None),
        )


@dataclass(frozen=True)
class CatalogueRecord:
    """One subtitle in the catalogue.

    An episode record names its series, not its episode.
    The catalogue puts the season and the episode at the end of a title, and both titles are stored without it.
    """

    #: Separates the release names inside one element
    RELEASE_SEPARATOR: ClassVar[str] = ';'

    catalogue_id: int

    #: Name of the series for an episode
    title: str | None

    #: The Polish title
    alternative_title: str | None

    #: Id of the series for an episode
    imdb_id: str | None

    year: int | None
    language: str | None

    #: Every release name that the subtitle was made for
    releases: tuple[str, ...]

    frame_rate: float | None
    season: int | None
    episode: int | None
    episode_title: str | None

    @classmethod
    def from_element(cls, element: RecordElement) -> CatalogueRecord | None:
        """Read one ``<subtitle>`` element.

        :param element: the element to read.
        :type element: RecordElement
        :return: the record, or None when the element has no id or holds a number that cannot be read.
        :rtype: CatalogueRecord | None

        """
        try:
            # int(None) raises a TypeError, so an element with no id lands in the except clause
            catalogue_id = int(element.text('id'))  # type: ignore[arg-type]
            year = element.integer('year')
            season = element.integer('season')
            episode = element.integer('episode')
            frame_rate = element.number('fps')

        except (TypeError, ValueError):
            logger.warning('Skipping a catalogue record that cannot be read')
            return None

        title_metadata = TitleMetadata.from_title(element.text('title'))
        alt_title_metadata = TitleMetadata.from_title(element.text('alttitle'))
        releases = element.text('release') or ''

        return cls(
            catalogue_id=catalogue_id,
            title=title_metadata.title,
            alternative_title=alt_title_metadata.title,
            imdb_id=element.text('imdb'),
            year=year,
            language=element.text('language'),
            releases=tuple(name for release in releases.split(cls.RELEASE_SEPARATOR) if (name := release.strip())),
            frame_rate=frame_rate,
            season=season if season is not None else title_metadata.season,
            episode=episode if episode is not None else title_metadata.episode,
            episode_title=element.text('eptitle'),
        )


@dataclass(frozen=True)
class CatalogueResponse:
    """The records that the catalogue search sent.

    The service sends at most 25 records, oldest first, and offers no way to ask for the rest.
    """

    #: Matches the XML declaration, which makes an HTML parser warn
    XML_DECLARATION_PATTERN: ClassVar[re.Pattern] = re.compile(rb'<\?xml.*?\?>')

    #: The whole body when the search found nothing
    EMPTY_BODY: ClassVar[bytes] = b'brak wynikow'

    #: The parsers to try, in order of preference
    PARSERS: ClassVar[tuple[str, ...]] = ('lxml', 'html.parser')

    records: tuple[CatalogueRecord, ...]

    @classmethod
    def from_response(cls, content: bytes) -> CatalogueResponse:
        """Read the response of the catalogue search.

        Parse the valid ``<subtitle>`` elements out of the XML-like response, and filter out the invalid ones.

        Napisy24 API does not respond with valid XML.
        It does not escape ``&`` characters, so a strict parser would stop at a title like ``Will & Grace``.
        We use a forgiving parser instead - ``lxml``, or ``html.parser`` when lxml is absent.
        ``html.parser`` warns about the XML declaration, so we strip the declaration out.

        It is a bit hacky, but that's the best we can do to parse a response that does not comply with the XML standard.

        :param bytes content: the bytes that the service sent.
        :return: the records, in the order the service sent them.
        :rtype: CatalogueResponse
        :raise: :class:`~subliminal.exceptions.ProviderError` if the body holds no ``<subtitle>`` element.

        """
        body = cls.XML_DECLARATION_PATTERN.sub(b'', content).strip()
        if body == cls.EMPTY_BODY:
            return cls(records=())

        soup = ParserBeautifulSoup(body, cls.PARSERS)
        elements = soup.find_all('subtitle')
        if not elements:
            msg = 'Cannot read the response of the catalogue search'
            raise ProviderError(msg)

        records = (CatalogueRecord.from_element(RecordElement(tag)) for tag in elements)
        return cls(records=tuple(record for record in records if record is not None))


# --------------------------------------------------------------------------------------------------
# Domain
#
# The rules that decide whether a subtitle belongs to a video.
# The domain knows the Video. It knows nothing about HTTP.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoIdentity:
    """Compares an IMDB id that the service sent with the IMDB id of the video.

    A hash match does not guarantee that the subtitle is for this video.
    Users have reported the service sending a subtitle for a different title, and only the IMDB id catches it.
    https://forum.napisy24.pl/viewtopic.php?f=9&t=142

    IMDB gives a separate id to a series and to each of its episodes.
    Game of Thrones is ``tt0944947``, and its episode S03E10 is ``tt2178796``.
    Both are stored on an ``Episode``, as ``imdb_id`` and ``series_imdb_id``.

    The hash search returns the id of the episode, and the catalogue search returns the id of the series.
    We compare against the one that the search returned, or a correct subtitle is refused.
    """

    video: Video

    def accepts_hash_response(self, response: HashResponse) -> bool:
        """Whether the response of the hash search is about this video.

        The hash search returns the id of the video itself, so we compare against ``imdb_id``.
        """
        return self._agrees(response.imdb_id, self.video.external_ids.get('imdb_id'))

    def accepts_catalogue_record(self, record: CatalogueRecord) -> bool:
        """Whether the catalogue record is about this video.

        The catalogue returns the id of the series for an episode, so we compare against ``series_imdb_id``.
        """
        video_imdb_id = (
            self.video.external_ids.get('series_imdb_id')
            if isinstance(self.video, Episode)
            else self.video.external_ids.get('imdb_id')
        )
        return self._agrees(record.imdb_id, video_imdb_id)

    @staticmethod
    def _agrees(subtitle_imdb_id: str | None, video_imdb_id: str | None) -> bool:
        """Whether two IMDB ids name the same title.

        An id that one side does not know is not a disagreement, so it passes.
        A stricter rule would throw away every subtitle for a video that no refiner could identify.

        The hash search sends bare digits, ``770828``, and the catalogue search sends the ``tt`` prefix, ``tt0770828``.
        Both sides are reduced to their digits before the comparison.
        """
        if subtitle_imdb_id is None or video_imdb_id is None:
            return True

        return sanitize_id(subtitle_imdb_id) == sanitize_id(video_imdb_id)


@dataclass(frozen=True)
class CatalogueQuery:
    """What to send to the catalogue search to find the subtitles of a video."""

    video: Video

    @property
    def parameters(self) -> dict[str, str]:
        """The request parameters of the catalogue search.

        The search takes an IMDB id or a title, and never both.

        - A movie is queried by its IMDB id and falls back to the title when a refiner can't find an id.
        - An episode is always queried by title, with the season and the episode appended.
          Example: ``title=Game of Thrones 3x10``

        We can't query an episode by IMDB id.
        The ``imdb`` parameter only accepts the id of a movie or a series.
        A query with the id of an episode returns nothing.
        A query with the id of the series returns multiple episodes instead.
        The problem is that the catalogue responds with just the 25 oldest records.
        It's not guaranteed that the episode we're looking for is there.
        """
        if isinstance(self.video, Episode):
            return {'title': f'{self.video.series} {self.video.season}x{self.video.episode}'}

        imdb_id = self.video.external_ids.get('imdb_id')
        if imdb_id is not None:
            return {'imdb': imdb_id}

        return {'title': str(self.video.title)}


@dataclass(frozen=True)
class HashMatcher:
    """Matches a subtitle against a video by the hash of the video file.

    The hash search returns no title and no release name, so the hash is the only match.
    """

    video_hash: str

    def get_matches(self, video: Video) -> set[str]:
        """Get the matches against the `video`."""
        return {'hash'} if video.hashes.get(PROVIDER_NAME) == self.video_hash else set()


@dataclass(frozen=True)
class CatalogueMatcher:
    """Matches a subtitle against a video by the data of its catalogue record.

    The catalogue search never reads the video file, so the hash is never a match.

    The catalogue keeps the original title and the Polish title, and subliminal takes one title at a time.
    Each title goes into its own guess, so a video named in Polish matches a record whose title is in English.
    """

    record: CatalogueRecord

    def get_matches(self, video: Video) -> set[str]:
        """Get the matches against the `video`."""
        metadata = {
            'year': self.record.year,
            'season': self.record.season,
            'episode': self.record.episode,
            'episode_title': self.record.episode_title,
            'fps': self.record.frame_rate,
        }
        titles = (self.record.title, self.record.alternative_title)
        video_type = 'episode' if isinstance(video, Episode) else 'movie'

        guesses = (
            *({**metadata, 'title': title} for title in titles if title is not None),
            *(safely_guessit(release, {'type': video_type}) for release in self.record.releases),
        )
        return {match for guess in guesses for match in guess_matches(video, guess)}


class Napisy24Subtitle(Subtitle):
    """A subtitle from Napisy24.

    A catalogue subtitle has an id, and the id maps to a page on the website.
    A pool subtitle has neither, so the hash of the video file identifies it instead.
    The id shows which of the two it is: ``catalogue_id:{catalogue_id}`` or ``hash:{video_hash}``.
    A hash belongs to the video file and not to the subtitle.
    That is weak, but a pool subtitle has nothing better.
    """

    provider_name: ClassVar[str] = PROVIDER_NAME

    #: Id of the subtitle in the catalogue, or zero for a pool subtitle
    catalogue_id: int

    def __init__(
        self,
        language: Language,
        matcher: SubtitleMatcher,
        *,
        catalogue_id: int,
        video_hash: str | None = None,
        fps: float | None = None,
    ) -> None:
        subtitle_id = f'catalogue_id:{catalogue_id}' if catalogue_id else f'hash:{video_hash}'
        page_link = f'https://napisy24.pl/download?napisId={catalogue_id}' if catalogue_id else None
        super().__init__(language, subtitle_id, page_link=page_link, fps=fps)
        self.catalogue_id = catalogue_id
        self.matcher = matcher

    @classmethod
    def from_hash_response(cls, language: Language, response: HashResponse, *, video_hash: str) -> Napisy24Subtitle:
        """Build the subtitle that the hash search returned.

        :param language: language of the subtitle.
        :type language: :class:`~babelfish.language.Language`
        :param response: the response of the hash search.
        :type response: HashResponse
        :param str video_hash: hash of the video file that the search used.
        :return: the subtitle.
        :rtype: Napisy24Subtitle

        """
        return cls(
            language,
            HashMatcher(video_hash),
            catalogue_id=response.catalogue_id,
            video_hash=video_hash,
            fps=response.frame_rate,
        )

    @classmethod
    def from_catalogue_record(cls, language: Language, record: CatalogueRecord) -> Napisy24Subtitle:
        """Build the subtitle from one record of the catalogue search.

        :param language: language of the subtitle.
        :type language: :class:`~babelfish.language.Language`
        :param record: the record to read.
        :type record: CatalogueRecord
        :return: the subtitle.
        :rtype: Napisy24Subtitle

        """
        return cls(
            language,
            CatalogueMatcher(record),
            catalogue_id=record.catalogue_id,
            fps=record.frame_rate,
        )

    def get_matches(self, video: Video) -> set[str]:
        """Get the matches against the `video`."""
        return self.matcher.get_matches(video)


# --------------------------------------------------------------------------------------------------
# Transport
#
# One function for each request the service takes, and each one returns bytes.
# The transport knows HTTP. It reads no text of the service, and it knows no Video.
# --------------------------------------------------------------------------------------------------


def check_response(response: Response) -> None:
    """Raise when the status of the response is not ``200``.

    Status ``503`` discards the provider for the rest of the run, and any other bad status fails one video only.

    :raise: :class:`~subliminal.exceptions.ServiceUnavailable` if the service is down,
        :class:`~subliminal.exceptions.ProviderError` for any other status that is not ``200``.

    """
    if response.status_code == HTTPStatus.SERVICE_UNAVAILABLE:
        msg = 'The service is unavailable'
        raise ServiceUnavailable(msg)

    if response.status_code != HTTPStatus.OK:
        msg = f'The service sent status {response.status_code}'
        raise ProviderError(msg)


def search_by_hash(
    session: Session,
    *,
    username: str,
    password: str,
    video_hash: str,
    size: int,
    name: str,
    timeout: int,
) -> bytes:
    """Search the catalogue and the program pool for the video file itself.

    :param int size: size of the video file, in bytes.
    :param str name: base name of the video file, and not its path.
    :param int timeout: seconds to wait for the service.
    :return: the bytes that the service sent.
    :rtype: bytes

    """
    # `n24pref=1` asks the service to prefer the catalogue copy when both collections hold the subtitle.
    # The service takes four more parameters, and none of them is ever sent:
    # - `md` is a second way to name the file, and it never works: a correct `md` with a wrong `fh` finds nothing.
    # - `nl` asks for one language, and the service ignores it.
    # - `licz` counts interest in a file, for the statistics of the service.
    #   Subliminal can ask for the same video more than once, and every request would add to the count.
    # - `tb` parameter, when present, regardless of its value, makes the service send a catalogue subtitle only.
    #   It then ignores the program pool.
    response = session.post(
        'http://napisy24.pl/run/CheckSubAgent.php',
        data={
            'postAction': 'CheckSub',
            'ua': username,
            'ap': password,
            'fh': video_hash,
            'fs': size,
            'fn': name,
            'n24pref': 1,
        },
        timeout=timeout,
    )
    check_response(response)
    return response.content


def search_catalogue(session: Session, *, parameters: dict[str, str], timeout: int) -> bytes:
    """Search the catalogue for the records of a video.

    The service takes an IMDB id or a title, and never both.
    An unknown parameter gives no error: it gives a well formed record for another title.

    :param dict parameters: the request parameters, which name either the IMDB id or the title.
    :param int timeout: seconds to wait for the service.
    :return: the bytes that the service sent.
    :rtype: bytes

    """
    response = session.get('http://napisy24.pl/libs/webapi.php', params=parameters, timeout=timeout)
    check_response(response)
    return response.content


def download_archive(session: Session, *, catalogue_id: int, timeout: int) -> bytes:
    """Download the archive that holds one catalogue subtitle.

    The service sends 500 for an id the catalogue does not hold, and a subtitle that was deleted is the likely reason.

    :param int catalogue_id: id of the subtitle in the catalogue.
    :param int timeout: seconds to wait for the service.
    :return: the ZIP archive that the service sent.
    :rtype: bytes

    """
    # typ asks for SubRip in UTF-8, and sru is the only value the service honours for every subtitle.
    # The Referer header is needed: without it the service would send a redirect and no archive.
    params: dict[str, Any] = {'napisId': catalogue_id, 'typ': 'sru'}
    response = session.get(
        'http://napisy24.pl/run/pages/download.php',
        params=params,
        headers={'Referer': 'http://napisy24.pl/'},
        timeout=timeout,
    )
    check_response(response)
    return response.content


# --------------------------------------------------------------------------------------------------
# Provider
#
# Makes a search, reads the response with a parser, and applies the rules.
# The provider knows HTTP and it knows the Video.
# --------------------------------------------------------------------------------------------------


class Napisy24Provider(Provider[Napisy24Subtitle]):
    """Napisy24 Provider.

    napisy24.pl needs an API username and password, and the provider comes with defaults, so credentials are optional.

    The only way to get API credentials is a PM to the napisy24.pl admin:
    https://forum.napisy24.pl/viewtopic.php?f=9&t=142
    Notice: a forum username and password is NOT the same as API credentials.

    :param str username: napisy24 API username (not mandatory)
    :param str password: napisy24 API password (not mandatory)
    :param int timeout: timeout in seconds. Default to 10.

    """

    languages: ClassVar[Set[Language]] = {Language('pol')}

    session: Session | None

    # The idea is that the credentials are bound to a program (like subliminal), not to a user of that program.
    # Other tools (Bazarr, Sub-Zero and Stremio addons) use them too.
    # They are technically a secret, but in reality they are public.
    # So, even though it feels odd, it's OK to hardcode these credentials in the provider source code.
    def __init__(
        self,
        username: str = 'subliminal',
        password: str = 'lanimilbus',  # noqa: S107
        timeout: int = 10,
    ) -> None:
        self.username = username
        self.password = password
        self.timeout = timeout
        self.session = None

    def initialize(self) -> None:
        """Open the session."""
        self.session = Session()
        self.session.headers['User-Agent'] = self.user_agent

    def terminate(self) -> None:
        """Close the session."""
        if self.session is None:
            raise NotInitializedProviderError

        self.session.close()
        self.session = None

    def query(self, video: Video, language: Language) -> list[Napisy24Subtitle]:
        """Search both collections for the subtitles of a video.

        The hash search runs first, because a hash match already scores the maximum.
        The catalogue search runs only when the hash search returns nothing.

        :param video: the video to search subtitles for.
        :type video: :class:`~subliminal.video.Video`
        :param language: language to assign to the subtitles.
        :type language: :class:`~babelfish.language.Language`
        :return: the subtitles found for the video.
        :rtype: list[Napisy24Subtitle]

        """
        if self.session is None:
            raise NotInitializedProviderError

        subtitle = self._search_by_hash(self.session, video, language)
        if subtitle is not None:
            return [subtitle]

        return self._search_catalogue(self.session, video, language)

    def list_subtitles(self, video: Video, languages: Set[Language]) -> list[Napisy24Subtitle]:
        """List all the subtitles for the video."""
        return [subtitle for language in languages for subtitle in self.query(video, language)]

    def download_subtitle(self, subtitle: Napisy24Subtitle) -> None:
        """Download the content of the subtitle.

        The hash search sends the metadata and the archive in one response.
        A subtitle it found already holds its content, so only a catalogue subtitle needs a request.
        """
        if self.session is None:
            raise NotInitializedProviderError

        if subtitle.content is not None:
            return

        archive = download_archive(self.session, catalogue_id=subtitle.catalogue_id, timeout=self.timeout)
        subtitle.set_content(read_archive(archive))

    def _search_by_hash(self, session: Session, video: Video, language: Language) -> Napisy24Subtitle | None:
        """Search both collections by the hash of the video file, and read the subtitle it returns."""
        video_hash = video.hashes.get(PROVIDER_NAME)
        if video_hash is None or video.size is None:
            return None

        content = search_by_hash(
            session,
            username=self.username,
            password=self.password,
            video_hash=video_hash,
            size=video.size,
            name=os.path.basename(video.name),
            timeout=self.timeout,
        )
        response = HashResponse.from_response(content)
        if response is None:
            return None

        if not VideoIdentity(video).accepts_hash_response(response):
            logger.warning('The hash search returned a subtitle for another title, IMDB id %s', response.imdb_id)
            return None

        subtitle_content = read_archive(response.archive)
        if subtitle_content is None:
            return None

        subtitle = Napisy24Subtitle.from_hash_response(language, response, video_hash=video_hash)
        subtitle.set_content(subtitle_content)
        return subtitle

    def _search_catalogue(self, session: Session, video: Video, language: Language) -> list[Napisy24Subtitle]:
        """Search the catalogue and build a subtitle for each record that matches the video.

        A record is dropped when its language is not the one asked for, or when its IMDB id names another title.
        A title query returns every title the service matched, so both filters are needed.
        """
        content = search_catalogue(session, parameters=CatalogueQuery(video).parameters, timeout=self.timeout)
        identity = VideoIdentity(video)

        return [
            Napisy24Subtitle.from_catalogue_record(language, record)
            for record in CatalogueResponse.from_response(content).records
            if record.language == language.alpha2 and identity.accepts_catalogue_record(record)
        ]
