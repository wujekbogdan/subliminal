from __future__ import annotations

import io
from pathlib import Path
from zipfile import ZipFile

import pytest

from subliminal.exceptions import AuthenticationError, ProviderError
from subliminal.providers.napisy24 import CatalogueRecord, CatalogueResponse, HashResponse, read_archive

DATA_DIR = Path(__file__).parent.parent / 'data' / 'napisy24'


def hash_response(name: str, archive: bytes = b'') -> bytes:
    """Build the response of the hash search from a data file.

    The data file holds one field on each line. The service sends the same fields on one line, with
    a ``|`` between them, then ``||``, then the archive. A field value must not hold a new line.

    :param str name: the name of the data file, without the extension.
    :param bytes archive: the archive that follows the header.
    :return: the bytes that the service sends.
    :rtype: bytes

    """
    lines = (DATA_DIR / f'{name}.txt').read_text(encoding='utf-8').splitlines()
    header = '|'.join(line for line in lines if line)
    return header.encode('utf-8') + b'||' + archive


def zip_archive(*members: tuple[str, bytes]) -> bytes:
    """Build a ZIP archive that holds the given members."""
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w') as archive:
        for member_name, content in members:
            archive.writestr(member_name, content)
    return buffer.getvalue()


class TestHashResponse:
    def test_reads_a_catalogue_subtitle(self) -> None:
        archive = zip_archive(
            ('Man.Of.Steel.2013.720p.BRRip.x264.AC3-EVO.srt', b'1\n00:00:49,591 --> 00:00:53,011\nx\n')
        )

        response = HashResponse.from_response(hash_response('hash_catalogue', archive))

        assert response == HashResponse(
            napisy_id=71928,
            imdb_id='tt0770828',
            frame_rate=23.976,
            archive=archive,
        )

    @pytest.mark.parametrize(
        'name',
        [
            pytest.param('hash_empty', id='OK-0, the service knows nothing'),
            pytest.param('hash_movie_only', id='OK-1, the movie is known, the subtitle is not'),
            pytest.param('hash_blocked', id='OK-3, the tb parameter stopped the subtitle'),
        ],
    )
    def test_is_none_when_no_subtitle_follows(self, name: str) -> None:
        assert HashResponse.from_response(hash_response(name)) is None

    def test_refuses_a_bad_account(self) -> None:
        with pytest.raises(AuthenticationError):
            HashResponse.from_response(b'login error')

    @pytest.mark.parametrize(
        'content',
        [
            pytest.param(b'', id='an empty body'),
            pytest.param(b'<html>Service unavailable</html>', id='not the format at all'),
            pytest.param(b'OK-2|fps:23.976|fimdb:770828|napisId:71928||', id='a subtitle is announced, none follows'),
            pytest.param(b'OK-2|fps:23.976|broken|napisId:71928||PK\x03\x04', id='a field with no name'),
            pytest.param(b'OK-2|fps:23.976|fimdb:770828||PK\x03\x04', id='the number of the subtitle is absent'),
            pytest.param(b'OK-2|fps:no|fimdb:770828|napisId:71928||PK\x03\x04', id='the frame rate is not a number'),
        ],
    )
    def test_refuses_an_unreadable_response(self, content: bytes) -> None:
        with pytest.raises(ProviderError):
            HashResponse.from_response(content)


class TestReadArchive:
    def test_picks_the_subtitle_and_not_the_shortcut(self) -> None:
        subtitle = b'1\n00:00:49,591 --> 00:00:53,011\nx\n'
        archive = zip_archive(
            ('Napisy24.pl.url', b'[InternetShortcut]\nURL=http://napisy24.pl/\n'),
            ('Dexter.S08E07.Dress.Code.720p.BluRay.DD5.1.x264-NTb.srt', subtitle),
        )

        assert read_archive(archive) == subtitle

    def test_is_none_when_no_member_is_a_subtitle(self) -> None:
        archive = zip_archive(('Napisy24.pl.url', b'[InternetShortcut]\nURL=http://napisy24.pl/\n'))

        assert read_archive(archive) is None

    def test_refuses_bytes_that_are_not_an_archive(self) -> None:
        with pytest.raises(ProviderError):
            read_archive(b'this is not a ZIP archive')


class TestCatalogueResponse:
    def test_reads_a_movie_record(self) -> None:
        records = CatalogueResponse.from_response((DATA_DIR / 'catalogue_movie.xml').read_bytes()).records

        assert records == (
            CatalogueRecord(
                napisy_id=71928,
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
        records = CatalogueResponse.from_response((DATA_DIR / 'catalogue_episode.xml').read_bytes()).records

        assert len(records) == 4
        assert records[2] == CatalogueRecord(
            napisy_id=69321,
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
        records = CatalogueResponse.from_response((DATA_DIR / 'catalogue_ampersand.xml').read_bytes()).records

        assert len(records) == 7
        assert records[0].title == 'Will & Grace'

    def test_is_empty_when_the_service_found_nothing(self) -> None:
        assert CatalogueResponse.from_response(b'brak wynikow').records == ()

    def test_drops_a_damaged_record_and_keeps_the_others(self) -> None:
        records = CatalogueResponse.from_response((DATA_DIR / 'catalogue_damaged.xml').read_bytes()).records

        assert [record.napisy_id for record in records] == [133224]
        assert records[0].frame_rate == 23.976  # the service wrote `23,976`
        assert records[0].season is None
        assert records[0].episode is None

    def test_refuses_an_unreadable_response(self) -> None:
        with pytest.raises(ProviderError):
            CatalogueResponse.from_response(b'<html><body>Service unavailable</body></html>')
