import json
import os
import re
import subprocess
from collections import namedtuple
import logging
import bisect

from .common import SushiError, get_extension
from . import chapters

MediaStreamInfo = namedtuple('MediaStreamInfo', ['id', 'info', 'default', 'title'])
SubtitlesStreamInfo = namedtuple('SubtitlesStreamInfo', ['id', 'info', 'type', 'default', 'title'])
MediaInfo = namedtuple('MediaInfo', ['video', 'audio', 'subtitles', 'chapters'])


class FFmpeg(object):
    @staticmethod
    def get_info_v2(path):
        try:
            args = [
                'ffprobe',
                '-v', 'quiet',
                '-show_streams',
                '-show_chapters',
                '-show_entries',
                'stream=index,codec_name,codec_type,sample_rate,width,height,channel_layout,bits_per_raw_sample,profile:'
                'stream_tags=title,language:',
                'stream_disposition=default,forced'
                'chapter=start_time',
                '-print_format', 'json=compact=1',
                path
            ]

            process = subprocess.Popen(
                args, 
                stdout=subprocess.PIPE,
                text=True, 
                encoding='utf-8'
            )
            output, _ = process.communicate()
            return output
        except OSError as e:
            if e.errno == 2:
                raise SushiError("Couldn't invoke ffprobe, check that it's installed")
            raise

    @staticmethod
    def demux_file(input_path, **kwargs):
        args = [
            'ffmpeg', 
            '-hide_banner', 
            '-loglevel', 'error', 
            '-stats', 
            '-i', input_path, 
            '-y'
        ]

        audio_stream = kwargs.get('audio_stream', None)
        audio_path = kwargs.get('audio_path', None)
        audio_rate = kwargs.get('audio_rate', None)
        if audio_stream is not None:
            args.extend(('-map', '0:{0}'.format(audio_stream)))
            if audio_rate:
                args.extend(('-ar', str(audio_rate)))
            args.extend(('-ac', '1', '-acodec', 'pcm_s16le', audio_path))

        script_stream = kwargs.get('script_stream', None)
        script_path = kwargs.get('script_path', None)
        if script_stream is not None:
            args.extend(('-map', '0:{0}'.format(script_stream), script_path))

        video_stream = kwargs.get('video_stream', None)
        timecodes_path = kwargs.get('timecodes_path', None)
        if timecodes_path is not None:
            args.extend(('-map', '0:{0}'.format(video_stream), '-f', 'mkvtimestamp_v2', timecodes_path))

        logging.info('ffmpeg args: {0}'.format(' '.join(('"{0}"' if ' ' in a else '{0}').format(a) for a in args)))
        try:
            subprocess.call(args)
        except OSError as e:
            if e.errno == 2:
                raise SushiError("Couldn't invoke ffmpeg, check that it's installed")
            raise

    @staticmethod
    def _get_audio_streams_v2(parsed_streams):
        streams = []
        for s in parsed_streams:
            idx = s.get('index')
            title = s.get('tags', {}).get('title')
            default = s.get('disposition', {}).get('default', 0) == 1

            codec = s.get('codec_name')
            channel_layout = s.get('channel_layout')

            sample_rate = f'{s.get("sample_rate")} Hz' if s.get("sample_rate") else None
            bit_depth = f'{s.get("bits_per_raw_sample")} bits' if s.get('bits_per_raw_sample') else None
            forced_flag = 'forced' if s.get('disposition', {}).get('forced', 0) == 1 else None

            additional_info = ', '.join(filter(None, [codec, sample_rate, channel_layout, bit_depth, forced_flag]))

            streams.append(MediaStreamInfo(idx, additional_info, default, title))
        
        return streams
    
    @staticmethod
    def _get_video_streams_v2(parsed_streams):
        streams = []
        for s in parsed_streams:
            idx = s.get('index')
            title = s.get('tags', {}).get('title', '')
            default = s.get('disposition', {}).get('default', 0) == 1

            codec = s.get('codec_name', '')
            codec_profile = s.get('profile', '')
            width = s.get('width')
            height = s.get('height')
            
            _resolution = f'{width}x{height}' if width and height else None
            _codec_info = f'{codec} ({codec_profile})' if codec_profile else codec

            additional_info = ', '.join(filter(None, [_codec_info, _resolution]))

            streams.append(MediaStreamInfo(idx, additional_info, default, title))
        
        return streams

    @staticmethod
    def _get_subtitles_streams_v2(parsed_streams):
        supported_formats = { 
            'ssa': '.ass',
            'ass': '.ass',
            'subrip': '.srt'
        }

        streams = []
        for s in parsed_streams:
            idx = s.get('index')
            title = s.get('tags', {}).get('title', '')
            default = s.get('disposition', {}).get('default', 0) == 1
            forced = s.get('disposition', {}).get('forced', 0) == 1

            language = s.get('tags', {}).get('language')
            sub_type = s.get("codec_name", '')
            sub_ext = supported_formats.get(sub_type)

            if (sub_ext is None):
               logging.warning(f"Unsupported subtitle format: {sub_type}. Skipping...")
               continue
            
            additional_info = f'{sub_type}'
            formatted_title = ' '.join(filter(None, [
                title, 
                f'({language})' if language else None,
                f'(forced)' if forced else None
            ]))

            streams.append(SubtitlesStreamInfo(idx, additional_info, sub_ext, default, formatted_title))
        
        return streams
    
    @staticmethod
    def _get_chapters_times_v2(chapters):
        return [float(c.get('start_time')) for c in chapters]

    @classmethod
    def get_media_info_v2(cls, path):
        info = cls.get_info_v2(path)
        streams, chapters = cls.get_clean_probe_info(info)

        video_streams = cls._get_video_streams_v2(streams.get('video', []))
        audio_streams = cls._get_audio_streams_v2(streams.get('audio', []))
        subs_streams = cls._get_subtitles_streams_v2(streams.get('subtitle', []))
        chapter_times = cls._get_chapters_times_v2(chapters)
        
        return MediaInfo(video_streams, audio_streams, subs_streams, chapter_times)

    @classmethod 
    def get_clean_probe_info(cls, info):
        try: 
            parsed = json.loads(info)
            streams_by_type = {}

            for stream in parsed["streams"]:
                codec_type = stream['codec_type']
                
                if codec_type == 'attachment':
                    continue

                streams_by_type.setdefault(codec_type, []).append(stream)

            return (streams_by_type, parsed["chapters"])
        except json.JSONDecodeError:
            raise SushiError("Couldn't parse ffprobe output, maybe it's too old?")
        except Exception as e:
            raise SushiError("Couldn't parse ffprobe output: {0}".format(str(e)))
        
class MkvToolnix(object):
    @classmethod
    def extract_timecodes(cls, mkv_path, stream_idx, output_path):
        args = ['mkvextract', 'timecodes_v2', mkv_path, '{0}:{1}'.format(stream_idx, output_path)]
        subprocess.call(args)


class SCXviD(object):
    @classmethod
    def make_keyframes(cls, video_path, log_path):
        try:
            ffmpeg_process = subprocess.Popen(['ffmpeg', '-i', video_path,
                                               '-f', 'yuv4mpegpipe',
                                               '-vf', 'scale=640:360',
                                               '-pix_fmt', 'yuv420p',
                                               '-vsync', 'drop', '-'],
                                              stdout=subprocess.PIPE)
        except OSError as e:
            if e.errno == 2:
                raise SushiError("Couldn't invoke ffmpeg, check that it's installed")
            raise

        try:
            scxvid_process = subprocess.Popen(['SCXvid', log_path], stdin=ffmpeg_process.stdout)
        except OSError as e:
            ffmpeg_process.kill()
            if e.errno == 2:
                raise SushiError("Couldn't invoke scxvid, check that it's installed")
            raise
        scxvid_process.wait()


class Timecodes(object):
    def __init__(self, times, default_fps):
        super(Timecodes, self).__init__()
        self.times = times
        self.default_frame_duration = 1.0 / default_fps if default_fps else None

    def get_frame_time(self, number):
        try:
            return self.times[number]
        except IndexError:
            if not self.default_frame_duration:
                return self.get_frame_time(len(self.times) - 1)
            if self.times:
                return self.times[-1] + (self.default_frame_duration) * (number - len(self.times) + 1)
            else:
                return number * self.default_frame_duration

    def get_frame_number(self, timestamp):
        if (not self.times or self.times[-1] < timestamp) and self.default_frame_duration:
            return int((timestamp - sum(self.times)) / self.default_frame_duration)
        return bisect.bisect_left(self.times, timestamp)

    def get_frame_size(self, timestamp):
        try:
            number = bisect.bisect_left(self.times, timestamp)
        except Exception:
            return self.default_frame_duration

        c = self.get_frame_time(number)

        if number == len(self.times):
            p = self.get_frame_time(number - 1)
            return c - p
        else:
            n = self.get_frame_time(number + 1)
            return n - c

    @classmethod
    def _convert_v1_to_v2(cls, default_fps, overrides):
        # start, end, fps
        overrides = [(int(x[0]), int(x[1]), float(x[2])) for x in overrides]
        if not overrides:
            return []

        fps = [default_fps] * (overrides[-1][1] + 1)
        for o in overrides:
            fps[o[0]:o[1] + 1] = [o[2]] * (o[1] - o[0] + 1)

        v2 = [0]
        for d in (1.0 / f for f in fps):
            v2.append(v2[-1] + d)
        return v2

    @classmethod
    def parse(cls, text):
        lines = text.splitlines()
        if not lines:
            return []
        first = lines[0].lower().lstrip()
        if first.startswith('# timecode format v2') or first.startswith('# timestamp format v2'):
            tcs = [float(x) / 1000.0 for x in lines[1:]]
            return Timecodes(tcs, None)
        elif first.startswith('# timecode format v1'):
            default = float(lines[1].lower().replace('assume ', ""))
            overrides = (x.split(',') for x in lines[2:])
            return Timecodes(cls._convert_v1_to_v2(default, overrides), default)
        else:
            raise SushiError('This timecodes format is not supported')

    @classmethod
    def from_file(cls, path):
        with open(path) as file:
            return cls.parse(file.read())

    @classmethod
    def cfr(cls, fps):
        class CfrTimecodes(object):
            def __init__(self, fps):
                self.frame_duration = 1.0 / fps

            def get_frame_time(self, number):
                return number * self.frame_duration

            def get_frame_size(self, timestamp):
                return self.frame_duration

            def get_frame_number(self, timestamp):
                return int(timestamp / self.frame_duration)

        return CfrTimecodes(fps)


class Demuxer(object):
    def __init__(self, path):
        super(Demuxer, self).__init__()
        self._path = path
        self._is_wav = get_extension(self._path) == '.wav'
        self._mi = None if self._is_wav else FFmpeg.get_media_info_v2(self._path)
        self._demux_audio = self._demux_subs = self._make_timecodes = self._make_keyframes = self._write_chapters = False

    @property
    def is_wav(self):
        return self._is_wav

    @property
    def path(self):
        return self._path

    @property
    def chapters(self):
        if self.is_wav:
            return []
        return self._mi.chapters

    @property
    def has_video(self):
        return not self.is_wav and self._mi.video

    def set_audio(self, stream_idx, output_path, sample_rate):
        self._audio_stream = self._select_stream(self._mi.audio, stream_idx, 'audio')
        self._audio_output_path = output_path
        self._audio_sample_rate = sample_rate
        self._demux_audio = True

    def set_script(self, stream_idx, output_path):
        self._script_stream = self._select_stream(self._mi.subtitles, stream_idx, 'subtitles')
        self._script_output_path = output_path
        self._demux_subs = True

    def set_timecodes(self, output_path):
        self._timecodes_output_path = output_path
        self._make_timecodes = True

    def set_chapters(self, output_path):
        self._write_chapters = True
        self._chapters_output_path = output_path

    def set_keyframes(self, output_path):
        self._keyframes_output_path = output_path
        self._make_keyframes = True

    def get_subs_type(self, stream_idx):
        return self._select_stream(self._mi.subtitles, stream_idx, 'subtitles').type

    def demux(self):
        if self._write_chapters:
            with open(self._chapters_output_path, "w") as output_file:
                output_file.write(chapters.format_ogm_chapters(self.chapters))

        if self._make_keyframes:
            SCXviD.make_keyframes(self._path, self._keyframes_output_path)

        ffargs = {}
        if self._demux_audio:
            ffargs['audio_stream'] = self._audio_stream.id
            ffargs['audio_path'] = self._audio_output_path
            ffargs['audio_rate'] = self._audio_sample_rate
        if self._demux_subs:
            ffargs['script_stream'] = self._script_stream.id
            ffargs['script_path'] = self._script_output_path

        if self._make_timecodes:
            def set_ffmpeg_timecodes():
                ffargs['video_stream'] = self._mi.video[0].id
                ffargs['timecodes_path'] = self._timecodes_output_path

            if get_extension(self._path).lower() == '.mkv':
                try:
                    MkvToolnix.extract_timecodes(self._path,
                                                 stream_idx=self._mi.video[0].id,
                                                 output_path=self._timecodes_output_path)
                except OSError as e:
                    if e.errno == 2:
                        set_ffmpeg_timecodes()
                    else:
                        raise
            else:
                set_ffmpeg_timecodes()

        if ffargs:
            FFmpeg.demux_file(self._path, **ffargs)

    def cleanup(self):
        if self._demux_audio:
            os.remove(self._audio_output_path)
        if self._demux_subs:
            os.remove(self._script_output_path)
        if self._make_timecodes:
            os.remove(self._timecodes_output_path)
        if self._write_chapters:
            os.remove(self._chapters_output_path)

    @classmethod
    def _format_stream(cls, stream):
        return '{0}{1}: {2}'.format(stream.id, ' (%s)' % stream.title if stream.title else '', stream.info)

    @classmethod
    def _format_streams_list(cls, streams):
        return '\n'.join(map(cls._format_stream, streams))

    def _select_stream(self, streams, chosen_idx, stream_type):
        formatted_stream_type = (stream_type[:-1] if stream_type.endswith('s') else stream_type)
        if not streams:
            raise SushiError('No {0} streams found in {1}'.format(formatted_stream_type, self._path))
        if chosen_idx is None:
            if len(streams) > 1:
                default_track = next((s for s in streams if s.default), None)
                if default_track:
                    logging.warning('Using default {0} track {1} in {2} because there are multiple candidates'
                                    .format(formatted_stream_type, self._format_stream(default_track), self._path))
                    return default_track
                raise SushiError('More than one {0} stream found in {1}.'
                                 'You need to specify the exact one to demux. Here are all candidates:\n'
                                 '{2}'.format(formatted_stream_type, self._path, self._format_streams_list(streams)))
            return streams[0]

        try:
            return next(x for x in streams if x.id == chosen_idx)
        except StopIteration:
            raise SushiError("{0} stream with index {1} doesn't exist in {2}.\n"
                             "Here are all that do:\n"
                             "{3}".format(formatted_stream_type.capitalize(), chosen_idx, self._path, self._format_streams_list(streams)))
