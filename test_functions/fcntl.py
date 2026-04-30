# Constants Metaflow needs to bypass the Linux check on Windows
F_SETFL = 0
F_GETFL = 0

def fcntl(fd, op, arg=0): return 0
def ioctl(fd, op, arg=0, mutable_flag=False): return 0
def flock(fd, op): return 0
def lockf(fd, op, length=0, start=0, whence=0): return 0