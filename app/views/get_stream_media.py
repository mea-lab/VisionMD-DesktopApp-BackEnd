# app/views/get_stream_media.py
import os, re, mimetypes

from django.conf import settings
from django.http import StreamingHttpResponse, Http404
from rest_framework.decorators import api_view

@api_view(['GET'])
def get_stream_media(request, path):
    fullpath = os.path.join(settings.MEDIA_ROOT, path)
    if not os.path.exists(fullpath):
        raise Http404

    size = os.path.getsize(fullpath)
    content_type, _ = mimetypes.guess_type(fullpath)
    content_type = content_type or 'application/octet-stream'

    range_header = request.META.get('HTTP_RANGE', '')
    range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
    if range_match:
        start = int(range_match.group(1))
        end   = int(range_match.group(2)) if range_match.group(2) else size - 1
        end   = min(end, size - 1)
        length = end - start + 1

        resp = StreamingHttpResponse(
            file_iterator(fullpath, offset=start, length=length),
            status=206,
            content_type=content_type
        )
        resp['Content-Range']   = f'bytes {start}-{end}/{size}'
        resp['Accept-Ranges']   = 'bytes'
        resp['Content-Length']  = str(length)
    else:
        # no Range header; return entire file
        resp = StreamingHttpResponse(
            file_iterator(fullpath),
            content_type=content_type
        )
        resp['Content-Length'] = str(size)
        resp['Accept-Ranges'] = 'bytes'

    return resp

def file_iterator(path, offset=0, length=None, chunk_size=8192):
    with open(path, 'rb') as f:
        f.seek(offset)
        remaining = length
        while True:
            read_len = chunk_size if remaining is None else min(remaining, chunk_size)
            data = f.read(read_len)
            if not data:
                break
            if remaining is not None:
                remaining -= len(data)
            yield data
