"""Skip-thought dataset preprocessor.

Every file in the input directory will be overwritten to a clean version where
each line is a sentence with letters, numbers, hyphens and/or apostrophes.

Usage: python preprocess.py [output_dir (defaults to ./books)]
"""

import os
import re
import string
import sys


def sentences(s):
  """Convert a string of text to lines of cleaned sentences."""
  sentences = s.split('.')
  lines = []
  for sentence in sentences:
    sentence = re.sub(r"[^A-Za-z0-9 '-]", " ", sentence)
    sentence = re.sub(r"[ ]+", " ", sentence)
    lines.append(sentence.strip())
  return '\n'.join(lines)


output_dir = './books'
if len(sys.argv) > 1:
  output_dir = sys.argv[1]

filenames = os.listdir(output_dir)
for filename in filenames:
  with open(os.path.join(output_dir, filename)) as f:
    contents = f.read()
  with open(os.path.join(output_dir, filename), 'w') as f:
    f.write(sentences(contents))

