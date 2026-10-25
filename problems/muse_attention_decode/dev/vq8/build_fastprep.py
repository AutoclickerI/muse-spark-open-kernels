"""Generate the cooperative-Q/K-prologue vq8 variant; preserve the consumer/producer and merge ABI."""
from pathlib import Path
import hashlib
HERE = Path(__file__).resolve().parent

def source():
    src = (HERE / 'vq8.cu').read_text()
    a = src.index('DI void prologue(')
    b = src.index('\ntemplate <int ST, bool MASK>', a)
    return src[:a] + (HERE / 'fastprep.cu.inc').read_text() + src[b:]

if __name__ == '__main__':
    src = source()
    path = HERE / 'out/vq8_fastprep.cu'
    path.write_text(src)
    print(path, hashlib.sha256(src.encode()).hexdigest())
