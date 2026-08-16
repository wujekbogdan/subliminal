from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile

import pytest
from babelfish import Language  # type: ignore[import-untyped]

from subliminal.exceptions import AuthenticationError, ProviderError
from subliminal.providers.napisy24 import (
    CatalogueQuery,
    CatalogueRecord,
    CatalogueResponse,
    HashResponse,
    Napisy24Subtitle,
    TitleMetadata,
    VideoIdentity,
    read_archive,
)

if TYPE_CHECKING:
    from subliminal.video import Episode, Movie


class ServiceResponse:
    """The bytes that Napisy24 sends, built from the recorded responses in ``tests/data/napisy24``."""

    DATA_DIR = Path(__file__).parent.parent / 'data' / 'napisy24'

    #: Separates the header from the archive
    RESPONSE_SEPARATOR = b'||'

    #: Separates the fields inside the header
    FIELD_SEPARATOR = '|'

    #: The four bytes that every ZIP file starts with, named after PKZIP
    ZIP_SIGNATURE = b'PK\x03\x04'

    @classmethod
    def hash_search(cls, name: str, archive: bytes = b'') -> bytes:
        """Build the response of the hash search from a recorded header.

        The recorded file holds one field on each line. The service sends the same fields on one
        line, with a ``|`` between them, then ``||``, then the archive. A field value never holds
        a new line, so the two forms carry the same data.
        """
        lines = (cls.DATA_DIR / f'{name}.txt').read_text(encoding='utf-8').splitlines()
        header = cls.FIELD_SEPARATOR.join(line for line in lines if line)
        return header.encode('utf-8') + cls.RESPONSE_SEPARATOR + archive

    @classmethod
    def damaged_hash_search(cls, header: str, archive: bytes = ZIP_SIGNATURE) -> bytes:
        """Build a hash search response whose header is written out in full, and is broken on purpose.

        The service is not known to send any of these. They are the ways our own reader can fail.
        """
        return header.encode('utf-8') + cls.RESPONSE_SEPARATOR + archive

    @classmethod
    def catalogue_search(cls, name: str) -> bytes:
        """Read the response of the catalogue search, exactly as the service sent it."""
        return (cls.DATA_DIR / f'{name}.xml').read_bytes()

    @staticmethod
    def archive(*members: tuple[str, bytes]) -> bytes:
        """Build a ZIP archive that holds the given members."""
        buffer = io.BytesIO()
        with ZipFile(buffer, 'w') as zip_file:
            for member_name, content in members:
                zip_file.writestr(member_name, content)
        return buffer.getvalue()


class TestHashResponse:
    def test_reads_a_catalogue_subtitle(self) -> None:
        archive = ServiceResponse.archive(
            ('Man.Of.Steel.2013.720p.BRRip.x264.AC3-EVO.srt', b'1\n00:00:49,591 --> 00:00:53,011\nx\n')
        )

        response = HashResponse.from_response(ServiceResponse.hash_search('hash_catalogue', archive))

        assert response == HashResponse(
            catalogue_id=71928,
            imdb_id='tt0770828',
            frame_rate=23.976,
            archive=archive,
        )

    def test_reads_a_subtitle_that_carries_no_metadata(self) -> None:
        archive = ServiceResponse.archive(('nativelog.txt', b'not a subtitle at all'))

        response = HashResponse.from_response(ServiceResponse.hash_search('hash_without_metadata', archive))

        assert response == HashResponse(catalogue_id=0, imdb_id=None, frame_rate=None, archive=archive)

    @pytest.mark.parametrize(
        'name',
        [
            pytest.param('hash_empty', id='OK-0, the service knows nothing'),
            pytest.param('hash_movie_only', id='OK-1, the movie is known, the subtitle is not'),
            pytest.param('hash_blocked', id='OK-3, the tb parameter stopped the subtitle'),
        ],
    )
    def test_is_none_when_no_subtitle_follows(self, name: str) -> None:
        assert HashResponse.from_response(ServiceResponse.hash_search(name)) is None

    def test_refuses_a_bad_account(self) -> None:
        with pytest.raises(AuthenticationError):
            HashResponse.from_response(b'login error')

    @pytest.mark.parametrize(
        'content',
        [
            pytest.param(b'', id='an empty body'),
            pytest.param(b'<html>Service unavailable</html>', id='not the format at all'),
            pytest.param(
                ServiceResponse.damaged_hash_search('OK-2|fps:23.976|fimdb:770828|napisId:71928', archive=b''),
                id='a subtitle is announced, none follows',
            ),
            pytest.param(
                ServiceResponse.damaged_hash_search('OK-2|fps:23.976|broken|napisId:71928'),
                id='a field with no name',
            ),
            pytest.param(
                ServiceResponse.damaged_hash_search('OK-2|fps:23.976|fimdb:770828'),
                id='the number of the subtitle is absent',
            ),
            pytest.param(
                ServiceResponse.damaged_hash_search('OK-2|fps:no|fimdb:770828|napisId:71928'),
                id='the frame rate is not a number',
            ),
        ],
    )
    def test_refuses_an_unreadable_response(self, content: bytes) -> None:
        with pytest.raises(ProviderError):
            HashResponse.from_response(content)


class TestReadArchive:
    def test_picks_the_subtitle_and_not_the_shortcut(self) -> None:
        subtitle = b'1\n00:00:49,591 --> 00:00:53,011\nx\n'
        archive = ServiceResponse.archive(
            ('Napisy24.pl.url', b'[InternetShortcut]\nURL=http://napisy24.pl/\n'),
            ('Dexter.S08E07.Dress.Code.720p.BluRay.DD5.1.x264-NTb.srt', subtitle),
        )

        assert read_archive(archive) == subtitle

    def test_is_none_when_no_member_is_a_subtitle(self) -> None:
        archive = ServiceResponse.archive(('Napisy24.pl.url', b'[InternetShortcut]\nURL=http://napisy24.pl/\n'))

        assert read_archive(archive) is None

    def test_refuses_bytes_that_are_not_an_archive(self) -> None:
        with pytest.raises(ProviderError):
            read_archive(b'this is not a ZIP archive')


class TestTitleMetadata:
    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [
            pytest.param('Game of Thrones 3x10', ('Game of Thrones', 3, 10, True), id='a series'),
            pytest.param('The Office: The Accountants 3x00', ('The Office: The Accountants', 3, 0, True), id='ep 0'),
            pytest.param('Will & Grace  9x01', ('Will & Grace', 9, 1, True), id='two spaces before the suffix'),
            pytest.param('Breaking Bad 1x01-05', ('Breaking Bad', 1, 1, True), id='a range of episodes'),
            pytest.param('Man of Steel', ('Man of Steel', None, None, False), id='a movie'),
            pytest.param('Flicka 2', ('Flicka 2', None, None, False), id='a movie whose title ends in a number'),
            pytest.param('Blade Runner 2049', ('Blade Runner 2049', None, None, False), id='a movie named by a year'),
            pytest.param('Fahrenheit 451', ('Fahrenheit 451', None, None, False), id='a movie named by a number'),
            pytest.param('300', ('300', None, None, False), id='a movie whose whole title is a number'),
            pytest.param('2x4', ('2x4', None, None, False), id='a movie whose whole title looks like a suffix'),
        ],
    )
    def test_splits_a_catalogue_title(self, raw: str, expected: tuple[str, int | None, int | None, bool]) -> None:
        parsed = TitleMetadata.from_title(raw)

        assert (parsed.title, parsed.season, parsed.episode, parsed.is_episode) == expected


class TestCatalogueResponse:
    def test_reads_a_movie_record(self) -> None:
        records = CatalogueResponse.from_response(ServiceResponse.catalogue_search('catalogue_movie')).records

        assert records == (
            CatalogueRecord(
                catalogue_id=71928,
                title='Man of Steel',
                alternative_title='Człowiek ze stali',
                imdb_id='tt0770828',
                year=2013,
                language='pl',
                releases=(
                    'BDRip.x264-Larceny',
                    '720p.BluRay.x264-Felony',
                    '1080p.BluRay.x264-SECTOR7',
                    '1080p.BRRip.x264-YIFY',
                    '720p.BRRip.x264-YIFY',
                    '720p.BRRip.XviD.AC3-ViSiON',
                    'BRRip.XviD.AC3-ETRG',
                    'BRRip.XviD.AC3-SANTi',
                    '720p.BRRip.x264.AC3-EVO',
                ),
                frame_rate=23.976,
                season=None,
                episode=None,
                episode_title=None,
            ),
        )

    def test_reads_an_episode_record(self) -> None:
        records = CatalogueResponse.from_response(ServiceResponse.catalogue_search('catalogue_episode')).records

        assert len(records) == 4
        assert records[2] == CatalogueRecord(
            catalogue_id=69321,
            title='Game of Thrones',  # the ` 3x10` that the catalogue adds is removed
            alternative_title='Gra o tron',
            imdb_id='tt0944947',  # the id of the series, not the id of the episode
            year=2011,
            language='pl',
            releases=(
                '720p.WEB-DL.DD5.1.AAC2.0.H.264-YFN',
                '1080p.WEB-DL.DD5.1.AAC2.0.H.264-YFN',
                '720p.WEB-DL.DD5.1.H.264-NTb',
                '1080p.WEB-DL.DD5.1.H.264-NTb',
            ),
            frame_rate=23.976,
            season=3,
            episode=10,
            episode_title='Mhysa',
        )
        assert records[3].episode_title is None  # the element is present but empty

    def test_reads_a_response_with_a_bare_ampersand(self) -> None:
        records = CatalogueResponse.from_response(ServiceResponse.catalogue_search('catalogue_ampersand')).records

        assert len(records) == 7
        assert records[0].title == 'Will & Grace'

    def test_reads_the_season_and_episode_from_the_title_when_the_elements_are_absent(self) -> None:
        records = CatalogueResponse.from_response(ServiceResponse.catalogue_search('catalogue_no_season')).records

        assert [(record.season, record.episode) for record in records] == [(1, 2), (1, 2), (1, 2)]

    def test_is_empty_when_the_service_found_nothing(self) -> None:
        assert CatalogueResponse.from_response(b'brak wynikow').records == ()

    def test_drops_a_damaged_record_and_keeps_the_others(self) -> None:
        records = CatalogueResponse.from_response(ServiceResponse.catalogue_search('catalogue_damaged')).records

        assert [record.catalogue_id for record in records] == [133224]
        assert records[0].frame_rate == 23.976  # the service wrote `23,976`
        assert records[0].season is None
        assert records[0].episode is None

    def test_refuses_an_unreadable_response(self) -> None:
        with pytest.raises(ProviderError):
            CatalogueResponse.from_response(b'<html><body>Service unavailable</body></html>')


#: A catalogue record with every field empty. Fill only the fields a test is about, with ``replace``.
EMPTY_RECORD = CatalogueRecord(
    catalogue_id=0,
    title=None,
    alternative_title=None,
    imdb_id=None,
    year=None,
    language=None,
    releases=(),
    frame_rate=None,
    season=None,
    episode=None,
    episode_title=None,
)


class TestVideoIdentity:
    @staticmethod
    def hash_response(imdb_id: str | None) -> HashResponse:
        """A hash search response that carries nothing but the IMDB id under test."""
        return HashResponse(catalogue_id=0, imdb_id=imdb_id, frame_rate=None, archive=b'')

    def test_accepts_a_catalogue_record_that_names_the_series_of_an_episode(
        self,
        episodes: dict[str, Episode],
    ) -> None:
        episode = episodes['got_s03e10']
        series_imdb_id = episode.external_ids['series_imdb_id']

        assert VideoIdentity(episode).accepts_catalogue_record(replace(EMPTY_RECORD, imdb_id=series_imdb_id)) is True

    def test_refuses_a_hash_response_that_names_another_title(self, movies: dict[str, Movie]) -> None:
        identity = VideoIdentity(movies['man_of_steel'])

        assert identity.accepts_hash_response(self.hash_response('tt0944947')) is False

    @pytest.mark.parametrize(
        ('video_name', 'subtitle_imdb_id'),
        [
            pytest.param('man_of_steel', None, id='the service does not know the title'),
            pytest.param('enders_game', 'tt0770828', id='no refiner gave the video an id'),
        ],
    )
    def test_accepts_an_id_that_one_side_does_not_know(
        self,
        movies: dict[str, Movie],
        video_name: str,
        subtitle_imdb_id: str | None,
    ) -> None:
        identity = VideoIdentity(movies[video_name])

        assert identity.accepts_hash_response(self.hash_response(subtitle_imdb_id)) is True


class TestCatalogueQuery:
    def test_asks_for_a_movie_by_imdb_id(self, movies: dict[str, Movie]) -> None:
        assert CatalogueQuery(movies['man_of_steel']).parameters == {'imdb': 'tt0770828'}

    def test_asks_for_a_movie_by_title_when_no_refiner_gave_it_an_id(self, movies: dict[str, Movie]) -> None:
        assert CatalogueQuery(movies['enders_game']).parameters == {'title': "Ender's Game"}

    def test_asks_for_an_episode_by_title_and_never_by_imdb_id(self, episodes: dict[str, Episode]) -> None:
        assert CatalogueQuery(episodes['got_s03e10']).parameters == {'title': 'Game of Thrones 3x10'}


class TestNapisy24Subtitle:
    def test_a_catalogue_subtitle_is_identified_by_its_catalogue_id(self) -> None:
        subtitle = Napisy24Subtitle(Language('pol'), catalogue_id=71928)

        assert subtitle.id == 'catalogue_id:71928'

    def test_a_pool_subtitle_is_identified_by_the_video_hash(self) -> None:
        subtitle = Napisy24Subtitle(Language('pol'), catalogue_id=0, video_hash='5b8f8f4e41ccb21e')

        assert subtitle.id == 'hash:5b8f8f4e41ccb21e'

    def test_a_catalogue_subtitle_links_to_its_page(self) -> None:
        subtitle = Napisy24Subtitle(Language('pol'), catalogue_id=71928)

        assert subtitle.page_link == 'https://napisy24.pl/download?napisId=71928'
