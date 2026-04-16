## Sushi-next
Automatic shifter for SRT and ASS subtitle based on audio streams.

*Python 3.13 fork of https://github.com/tp7/Sushi.*

Credits to [DYY-Studio](https://github.com/DYY-Studio/Sushi) for the initial v3.13 porting work.

### Purpose
Imagine you've got a subtitle file synced to one video file, but you want to use these subtitles with some other video you've got via totally legal means. The common example is TV vs. BD releases, PAL vs. NTSC video and releases in different countries. In a lot of cases, subtitles won't match right away and you need to sync them.

The purpose of this script is to avoid all the hassle of manual syncing. It attempts to synchronize subtitles by finding similarities in audio streams. The script is very fast and can be used right when you want to watch something.

### How it works
You need to provide two audio files and a subtitle file that matches one of those files. For every line of the subtitles, the script will extract corresponding audio from the source audio stream and will try to find the closest similar pattern in the destination audio stream, obtaining a shift value which is later applied to the subtitles.

Detailed explanation of Sushi workflow and description of command-line arguments can be found in the [wiki][2].

### Usage
The minimal command line looks like this:
```
python -m sushi --src hdtv.wav --dst bluray.wav --script subs.ass
```
Output file name is optional - `"{destination_path}.sushi.{subtitles_format}"` is used by default. See the [usage][3] page of the wiki for further examples.

Do note that WAV is not the only format Sushi can work with. It can process audio/video files directly and decode various audio formats, provided that ffmpeg is available. For additional info refer to the [Demuxing][4] part of the wiki.

### Requirements
Sushi should work on Windows, Linux and OS X. Please open an issue if it doesn't. To run it, you have to have the following installed:

1. [Python 3.13 or higher][5]
2. [NumPy][6] (2.3.4 or newer)
3. [SciPy][6] (1.16.2 or newer)
4. [OpenCV 4.4.x or newer][7]
5. [FFmpeg][9] (for any kind of demuxing)

Optionally, you might want:
1. [MkvExtract][10] for faster timecodes extraction when demuxing
2. [SCXvid-standalone][11] if you want Sushi to make keyframes
3. [Colorama](https://github.com/tartley/colorama) to add colors to console output on Windows

### Development setup (uv)

Use [uv](https://docs.astral.sh/uv/) to create a local environment and install dependencies from `pyproject.toml`:

```bash
uv sync
```

Run Sushi from the managed environment:

```bash
uv run sushi --src hdtv.wav --dst bluray.wav --script subs.ass
```

Run tests:

```bash
uv run python run-tests.py
```

#### Installation on Windows
1. Install Python (64 bit).
2. Install FFmpeg.
2. Install OpenCV.
3. Run `pip install sushi-sub-next` on a terminal.
4. Use it as `sushi args…` on a terminal.

If anyone wants to provide proper installation steps or a binary for Windows, please open a PR or get in contact.

#### Installation on Mac OS X

No binary packages are provided for OS X right now so you'll have to use the script form. Assuming you have Python, pip and [homebrew](http://brew.sh/) installed, run the following:
```bash
brew install git opencv ffmpeg
pip3 install numpy
# install some optional dependencies
brew install f mkvtoolnix
# install sushi
pip install sushi-sub-next
# use sushi
sushi args…
```

#### Installation on Linux

If you have apt-get available, the installation process is trivial.
```bash
sudo apt-get update
sudo apt-get install git python3 python3-numpy python3-opencv ffmpeg

pip install --user sushi-sub-next
# if ~/.local/bin is in your PATH
sushi args…
# otherwise
python -m sushi args…
```

For other distros, pick corresponding package names for the python, numpy, and opencv dependencies.

### Limitations
This script will never be able to property handle frame-by-frame typesetting. If underlying video stream changes (e.g. has different telecine pattern), you might get incorrect output.

This script cannot improve bad timing. If original lines are mistimed, they will be mistimed in the output file too.

In short, while this might be safe for immediate viewing, you probably shouldn't use it to blindly shift subtitles for permanent storing.


  <!-- [1]: https://github.com/tp7/Sushi/releases -->
  [2]: https://github.com/tp7/Sushi/wiki
  [3]: https://github.com/tp7/Sushi/wiki/Examples
  [4]: https://github.com/tp7/Sushi/wiki/Demuxing
  [5]: https://www.python.org/downloads/
  [6]: http://www.scipy.org/scipylib/download.html
  [7]: http://opencv.org/
  [9]: http://www.ffmpeg.org/download.html
  [10]: http://www.bunkus.org/videotools/mkvtoolnix/downloads.html
  [11]: https://github.com/soyokaze/SCXvid-standalone/releases
