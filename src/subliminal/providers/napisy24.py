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
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar
from zipfile import BadZipFile, ZipFile

from subliminal.exceptions import AuthenticationError, ProviderError
from subliminal.subtitle import SUBTITLE_EXTENSIONS, Subtitle
from subliminal.utils import decorate_imdb_id, ensure_list, safely_guessit, sanitize_id
from subliminal.video import Episode

from . import ParserBeautifulSoup

if TYPE_CHECKING:
    from babelfish import Language  # type: ignore[import-untyped]
    from bs4 import Tag

    from subliminal.video import Video

logger = logging.getLogger(__name__)


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
    #: This provider never sends ``tb``, so the service never sends this status back.
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

    #: Id of the subtitle in the catalogue, or zero when the subtitle maps to no page on the website
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

    The archive can hold a ``Napisy24.pl.url`` shortcut next to the subtitle, so the member is chosen by extension.
    The first member of the archive is not always the subtitle.

    :param bytes archive: the ZIP archive.
    :return: the content of the subtitle, or None when no member is a subtitle.
    :rtype: bytes | None
    :raise: :class:`~subliminal.exceptions.ProviderError` if the bytes are not a ZIP archive.

    """
    try:
        with ZipFile(io.BytesIO(archive)) as zip_file:
            names = [name for name in zip_file.namelist() if name.lower().endswith(SUBTITLE_EXTENSIONS)]
            if not names:
                logger.warning('No subtitle in the archive, it holds these files: %r', zip_file.namelist())
                return None

            return zip_file.read(names[0])

    except BadZipFile as error:
        msg = 'The service sent bytes that are not a ZIP archive'
        raise ProviderError(msg) from error


@dataclass(frozen=True)
class RecordElement:
    """One ``<subtitle>`` element of a catalogue response.

    A method gives None when the child element is absent, and also when the child element holds no text.
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

    The title of an episode contains a season and an episode suffix, e.g.: ``Game of Thrones 3x10``.
    A movie title carries a name only.
    """

    #: Matches the suffix that the catalogue appends
    EPISODE_TOKEN: ClassVar[re.Pattern] = re.compile(r'\s\d{1,2}x\d{1,3}\b')

    #: The title, with the season and the episode removed
    title: str | None

    #: Season number, or None when the title carries none
    season: int | None

    #: The episode number, or None when the title carries none
    episode: int | None

    @property
    def is_episode(self) -> bool:
        """Whether the title names an episode of a series."""
        return self.season is not None

    @classmethod
    def from_title(cls, title: str | None) -> TitleMetadata:
        """Split a title into its parts."""
        # Parse it with guessit only if the title ends with an episode suffix.
        # Otherwise the title is considered a movie title that does not require any parsing.
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
    :attr:`title` holds the name of the series, and :attr:`imdb_id` holds the id of the series.
    The catalogue puts the season and the episode at the end of a title, and both titles are stored without it.
    """

    #: Separates the release names inside one element
    RELEASE_SEPARATOR: ClassVar[str] = ';'

    catalogue_id: int
    title: str | None
    alternative_title: str | None
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

        Napisy24 API does not respond with a valid XML.
        It does not escape ``&`` characters, so a strict parser would stop at a title like ``Will & Grace``.
        We use a forgiving parser instead - ``html.parser``.
        ``html.parser`` warns about the XML header, so we strip the header out.

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
# This section knows the Video. It knows nothing about HTTP.
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


class Napisy24Subtitle(Subtitle):
    """A subtitle from Napisy24.

    A catalogue subtitle has a catalogue id. A program pool subtitle has none.
    The id shows which of the two it is: ``catalogue_id:71928`` or ``hash:5b8f8f4e41ccb21e``.
    A hash identifies the video file and not the subtitle, so it is weak, but a pool subtitle has nothing better.
    """

    provider_name: ClassVar[str] = 'napisy24'

    #: Page of a catalogue subtitle. The service shows it to a signed-in reader only.
    PAGE_URL: ClassVar[str] = 'https://napisy24.pl/download?napisId={catalogue_id}'

    def __init__(self, language: Language, *, catalogue_id: int, video_hash: str | None = None) -> None:
        subtitle_id = f'catalogue_id:{catalogue_id}' if catalogue_id else f'hash:{video_hash}'
        page_link = self.PAGE_URL.format(catalogue_id=catalogue_id) if catalogue_id else None
        super().__init__(language, subtitle_id, page_link=page_link)
