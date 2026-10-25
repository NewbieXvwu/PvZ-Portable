"""Roof-only state comparison for the frozen engine's uninitialised cloud counters.

Challenge::Challenge leaves mCloudsCounter[6] unset. The only reads are in
TreeOfWisdomUpdate, outside the task76 adventure levels. Keep every other byte,
including RNG and counters; retain the original images and native hashes.
"""
import hashlib
from pathlib import Path
import struct
import zlib


def canonical_snapshot(data,level):
    assert 42<=level<=49, 'normalisation is restricted to task76 roof levels'
    assert data[:7]==b'PVZSB1\n'
    board_size=struct.unpack_from('<Q',data,7)[0]
    rand_offset=15+board_size
    rand_size=struct.unpack_from('<Q',data,rand_offset)[0]
    assert len(data)==rand_offset+8+rand_size+36, 'incomplete state/RNG image'
    board=bytearray(data[15:15+board_size])
    magic,version,payload_size,crc=struct.unpack_from('<12sIII',board)
    assert magic==b'PVZP_SAVE4\x00\x00' and version==1
    assert payload_size==len(board)-24 and zlib.crc32(board[24:])==crc
    offset=24;cloud_offset=None
    seen=set()
    while offset<len(board):
        tag,size=struct.unpack_from('<II',board,offset)
        assert tag not in seen and offset+8+size<=len(board)
        seen.add(tag);start=offset+8
        if tag==19:
            # Chunk version, wrapper field, object blob, tail field. Reject a
            # different format instead of masking an approximate byte range.
            assert size==372
            assert struct.unpack_from('<6I',board,start)==(1,1,360,356,100,348)
            cloud_offset=start+24+304
        offset=start+size
    assert offset==len(board) and seen==set(range(1,21)) and cloud_offset is not None
    counters=list(struct.unpack_from('<6i',board,cloud_offset))
    board[cloud_offset:cloud_offset+24]=bytes(24)
    struct.pack_into('<I',board,20,zlib.crc32(board[24:]))
    normal=data[:15]+bytes(board)+data[15+board_size:]
    metadata=dict(raw_sha256=hashlib.sha256(data).hexdigest(),
                  canonical_sha256=hashlib.sha256(normal).hexdigest(),
                  cloud_counters=counters,board_bytes=board_size,
                  cloud_bytes_offset=15+cloud_offset,normalised_cloud_bytes=24,
                  method='Only Challenge.mCloudsCounter[6] and the dependent SaveGame CRC; roof adventure only')
    return normal,metadata


def verify_snapshot(metadata,level):
    normal,actual=canonical_snapshot(Path(metadata['path']).read_bytes(),level)
    assert all(metadata[k]==v for k,v in actual.items()), 'state-image evidence changed'
    return normal
