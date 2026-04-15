import json
import subprocess
import unittest
from unittest import mock

from sushi.demux import FFmpeg, MkvToolnix, SCXviD
from sushi.common import SushiError
from sushi import chapters


def create_popen_mock():
    popen_mock = mock.Mock()
    process_mock = mock.Mock()
    process_mock.configure_mock(**{'communicate.return_value': ('ouput', 'error')})
    popen_mock.return_value = process_mock
    return popen_mock


class FFmpegTestCase(unittest.TestCase):
    ffprobe_info_json = {
        'streams': [
            {
                'index': 0,
                'codec_type': 'video',
                'codec_name': 'h264',
                'profile': 'High 10',
                'width': 1280,
                'height': 720,
                'disposition': {'default': 1},
                'tags': {'title': 'Video 10bit'}
            },
            {
                'index': 1,
                'codec_type': 'audio',
                'codec_name': 'aac',
                'sample_rate': '48000',
                'channel_layout': 'stereo',
                'bits_per_raw_sample': '16',
                'disposition': {'default': 1},
                'tags': {'title': 'Audio AAC 2.0'}
            },
            {
                'index': 2,
                'codec_type': 'audio',
                'codec_name': 'aac',
                'sample_rate': '48000',
                'channel_layout': 'stereo',
                'disposition': {'default': 0},
                'tags': {'title': 'English Audio AAC 2.0'}
            },
            {
                'index': 3,
                'codec_type': 'subtitle',
                'codec_name': 'ssa',
                'disposition': {'default': 1},
                'tags': {'language': 'eng', 'title': 'English Subtitles'}
            },
            {
                'index': 4,
                'codec_type': 'subtitle',
                'codec_name': 'subrip',
                'disposition': {'default': 0},
                'tags': {'language': 'enm', 'title': 'English (JP honorifics)'}
            },
            {
                'index': 5,
                'codec_type': 'attachment',
                'codec_name': 'ttf'
            }
        ],
        'chapters': [
            {'start_time': '0.000000'},
            {'start_time': '17.017000'},
            {'start_time': '107.023000'}
        ]
    }

    def test_get_clean_probe_info_filters_attachment_streams(self):
        streams_by_type, chapter_list = FFmpeg.get_clean_probe_info(json.dumps(self.ffprobe_info_json))

        self.assertEqual(len(streams_by_type['video']), 1)
        self.assertEqual(len(streams_by_type['audio']), 2)
        self.assertEqual(len(streams_by_type['subtitle']), 2)
        self.assertNotIn('attachment', streams_by_type)
        self.assertEqual(chapter_list, self.ffprobe_info_json['chapters'])

    def test_parses_audio_stream_v2(self):
        parsed_streams = [s for s in self.ffprobe_info_json['streams'] if s['codec_type'] == 'audio']
        audio = FFmpeg._get_audio_streams_v2(parsed_streams)

        self.assertEqual(len(audio), 2)
        self.assertEqual(audio[0].id, 1)
        self.assertEqual(audio[0].info, 'aac, 48000 Hz, stereo, 16 bits')
        self.assertTrue(audio[0].default)
        self.assertEqual(audio[0].title, 'Audio AAC 2.0')
        self.assertEqual(audio[1].id, 2)
        self.assertEqual(audio[1].info, 'aac, 48000 Hz, stereo')
        self.assertFalse(audio[1].default)
        self.assertEqual(audio[1].title, 'English Audio AAC 2.0')

    def test_parses_video_stream_v2(self):
        parsed_streams = [s for s in self.ffprobe_info_json['streams'] if s['codec_type'] == 'video']
        video = FFmpeg._get_video_streams_v2(parsed_streams)

        self.assertEqual(len(video), 1)
        self.assertEqual(video[0].id, 0)
        self.assertEqual(video[0].info, 'h264 (High 10), 1280x720')
        self.assertTrue(video[0].default)
        self.assertEqual(video[0].title, 'Video 10bit')

    def test_parses_subtitles_stream_v2(self):
        parsed_streams = [s for s in self.ffprobe_info_json['streams'] if s['codec_type'] == 'subtitle']
        subs = FFmpeg._get_subtitles_streams_v2(parsed_streams)

        self.assertEqual(len(subs), 2)
        self.assertEqual(subs[0].id, 3)
        self.assertEqual(subs[0].info, 'ssa')
        self.assertEqual(subs[0].type, '.ass')
        self.assertTrue(subs[0].default)
        self.assertEqual(subs[0].title, 'English Subtitles (eng)')
        self.assertEqual(subs[1].id, 4)
        self.assertEqual(subs[1].info, 'subrip')
        self.assertEqual(subs[1].type, '.srt')
        self.assertFalse(subs[1].default)
        self.assertEqual(subs[1].title, 'English (JP honorifics) (enm)')

    def test_parses_chapter_times_v2(self):
        chapter_times = FFmpeg._get_chapters_times_v2(self.ffprobe_info_json['chapters'])
        self.assertEqual(chapter_times, [0.0, 17.017, 107.023])

    @mock.patch('sushi.demux.logging.warning')
    def test_subtitles_stream_skips_unsupported_format(self, warning_mock):
        parsed_streams = [{
            'index': 7,
            'codec_name': 'dvd_subtitle',
            'tags': {'language': 'eng', 'title': 'PGS'},
            'disposition': {'default': 0}
        }]

        subtitles = FFmpeg._get_subtitles_streams_v2(parsed_streams)
        self.assertEqual(subtitles, [])
        warning_mock.assert_called_once_with('Unsupported subtitle format: dvd_subtitle. Skipping...')

    @mock.patch('subprocess.Popen', new_callable=create_popen_mock)
    def test_get_info_v2_call_args(self, popen_mock):
        self.assertEqual(FFmpeg.get_info_v2('random_file.mkv'), 'ouput')
        self.assertEqual(popen_mock.call_args[0][0], [
            'ffprobe',
            '-v', 'quiet',
            '-show_streams',
            '-show_chapters',
            '-show_entries', 'chapter=start_time',
            '-print_format', 'json=compact=1',
            'random_file.mkv'
        ])
        self.assertEqual(popen_mock.call_args[1], {
            'stdout': subprocess.PIPE,
            'text': True,
            'encoding': 'utf-8'
        })

    @mock.patch('subprocess.Popen', new_callable=create_popen_mock)
    def test_get_info_v2_fail_when_no_ffprobe(self, popen_mock):
        popen_mock.side_effect = OSError(2, 'ignored')
        self.assertRaises(SushiError, lambda: FFmpeg.get_info_v2('random.mkv'))

    @mock.patch.object(FFmpeg, 'get_info_v2')
    def test_get_media_info_v2(self, get_info_mock):
        get_info_mock.return_value = json.dumps(self.ffprobe_info_json)

        media_info = FFmpeg.get_media_info_v2('random.mkv')

        self.assertEqual(len(media_info.video), 1)
        self.assertEqual(len(media_info.audio), 2)
        self.assertEqual(len(media_info.subtitles), 2)
        self.assertEqual(media_info.chapters, [0.0, 17.017, 107.023])

    @mock.patch('subprocess.call')
    def test_demux_file_call_args(self, call_mock):
        FFmpeg.demux_file('random.mkv', audio_stream=0, audio_path='audio1.wav')
        FFmpeg.demux_file('random.mkv', audio_stream=0, audio_path='audio2.wav', audio_rate=12000)
        FFmpeg.demux_file('random.mkv', script_stream=0, script_path='subs1.ass')
        FFmpeg.demux_file('random.mkv', video_stream=0, timecodes_path='tcs1.txt')

        FFmpeg.demux_file('random.mkv', audio_stream=1, audio_path='audio0.wav', audio_rate=12000,
                          script_stream=2, script_path='out0.ass', video_stream=0, timecodes_path='tcs0.txt')

        call_mock.assert_any_call(['ffmpeg', '-hide_banner', '-i', 'random.mkv', '-y',
                                   '-map', '0:0', '-ac', '1', '-acodec', 'pcm_s16le', 'audio1.wav'])
        call_mock.assert_any_call(['ffmpeg', '-hide_banner', '-i', 'random.mkv', '-y',
                                   '-map', '0:0', '-ar', '12000', '-ac', '1', '-acodec', 'pcm_s16le', 'audio2.wav'])
        call_mock.assert_any_call(['ffmpeg', '-hide_banner', '-i', 'random.mkv', '-y',
                                   '-map', '0:0', 'subs1.ass'])
        call_mock.assert_any_call(['ffmpeg', '-hide_banner', '-i', 'random.mkv', '-y',
                                   '-map', '0:0', '-f', 'mkvtimestamp_v2', 'tcs1.txt'])
        call_mock.assert_any_call(['ffmpeg', '-hide_banner', '-i', 'random.mkv', '-y',
                                   '-map', '0:1', '-ar', '12000', '-ac', '1', '-acodec', 'pcm_s16le', 'audio0.wav',
                                   '-map', '0:2', 'out0.ass',
                                   '-map', '0:0', '-f', 'mkvtimestamp_v2', 'tcs0.txt'])


class MkvExtractTestCase(unittest.TestCase):
    @mock.patch('subprocess.call')
    def test_extract_timecodes(self, call_mock):
        MkvToolnix.extract_timecodes('video.mkv', 1, 'timecodes.tsc')
        call_mock.assert_called_once_with(['mkvextract', 'timecodes_v2', 'video.mkv', '1:timecodes.tsc'])


class SCXviDTestCase(unittest.TestCase):
    @mock.patch('subprocess.Popen')
    def test_make_keyframes(self, popen_mock):
        SCXviD.make_keyframes('video.mkv', 'keyframes.txt')
        self.assertTrue('ffmpeg' in (x.lower() for x in popen_mock.call_args_list[0][0][0]))
        self.assertTrue('scxvid' in (x.lower() for x in popen_mock.call_args_list[1][0][0]))

    @mock.patch('subprocess.Popen')
    def test_no_ffmpeg(self, popen_mock):
        def raise_no_app(cmd_args, **kwargs):
            if 'ffmpeg' in (x.lower() for x in cmd_args):
                raise OSError(2, 'ignored')

        popen_mock.side_effect = raise_no_app
        self.assertRaisesRegex(SushiError, '[fF][fF][mM][pP][eE][gG]',
                               lambda: SCXviD.make_keyframes('video.mkv', 'keyframes.txt'))

    @mock.patch('subprocess.Popen')
    def test_no_scxvid(self, popen_mock):
        def raise_no_app(cmd_args, **kwargs):
            if 'scxvid' in (x.lower() for x in cmd_args):
                raise OSError(2, 'ignored')
            return mock.Mock()

        popen_mock.side_effect = raise_no_app
        self.assertRaisesRegex(SushiError, '[sS][cC][xX][vV][iI][dD]',
                               lambda: SCXviD.make_keyframes('video.mkv', 'keyframes.txt'))


class ExternalChaptersTestCase(unittest.TestCase):
    def test_parse_xml_start_times(self):
        file_text = """<?xml version="1.0"?>
<!-- <!DOCTYPE Chapters SYSTEM "matroskachapters.dtd"> -->
<Chapters>
  <EditionEntry>
    <EditionUID>2092209815</EditionUID>
    <ChapterAtom>
      <ChapterUID>3122448259</ChapterUID>
      <ChapterTimeStart>00:00:00.000000000</ChapterTimeStart>
      <ChapterDisplay>
        <ChapterString>Prologue</ChapterString>
      </ChapterDisplay>
    </ChapterAtom>
    <ChapterAtom>
      <ChapterUID>998777246</ChapterUID>
      <ChapterTimeStart>00:00:17.017000000</ChapterTimeStart>
      <ChapterDisplay>
        <ChapterString>Opening Song ("YES!")</ChapterString>
      </ChapterDisplay>
    </ChapterAtom>
    <ChapterAtom>
      <ChapterUID>55571857</ChapterUID>
      <ChapterTimeStart>00:01:47.023000000</ChapterTimeStart>
      <ChapterDisplay>
        <ChapterString>Part A (Tale of the Doggypus)</ChapterString>
      </ChapterDisplay>
    </ChapterAtom>
  </EditionEntry>
</Chapters>
"""
        parsed_times = chapters.parse_xml_start_times(file_text)
        self.assertEqual(parsed_times, [0, 17.017, 107.023])

    def test_parse_ogm_start_times(self):
        file_text = """CHAPTER01=00:00:00.000
CHAPTER01NAME=Prologue
CHAPTER02=00:00:17.017
CHAPTER02NAME=Opening Song ("YES!")
CHAPTER03=00:01:47.023
CHAPTER03NAME=Part A (Tale of the Doggypus)
"""
        parsed_times = chapters.parse_ogm_start_times(file_text)
        self.assertEqual(parsed_times, [0, 17.017, 107.023])

    def test_format_ogm_chapters(self):
        chapters_text = chapters.format_ogm_chapters(start_times=[0, 17.017, 107.023])
        self.assertEqual(chapters_text, """CHAPTER01=00:00:00.000
CHAPTER01NAME=
CHAPTER02=00:00:17.017
CHAPTER02NAME=
CHAPTER03=00:01:47.023
CHAPTER03NAME=
""")
