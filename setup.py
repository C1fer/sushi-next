import io, os, re
from setuptools import setup

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# 1) 读取 __init__.py
here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, "sushi", "__init__.py"), encoding="utf-8") as f:
    content = f.read()

# 2) 正则提取 VERSION
VERSION = re.search(r"^VERSION\s*=\s*['\"]([^'\"]+)['\"]", content, re.M).group(1)

# 3) 读取 README
with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='sushi-sub',
    description='Automatic subtitle shifter based on audio',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['sushi'],
    version=VERSION,
    url='https://github.com/FichteFoll/Sushi',
    project_urls={
        'Documentation': 'https://github.com/tp7/Sushi/wiki',
        'Fork Origin': 'https://github.com/tp7/Sushi',
    },
    license='MIT',
    python_requires='>=3.5',
    install_requires=['numpy>=1.8'],
    entry_points={
        'console_scripts': [
            "sushi=sushi.__main__:main",
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Text Processing',
    ]
)
