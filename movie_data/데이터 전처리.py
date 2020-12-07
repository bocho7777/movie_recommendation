!pip install pytorch_pretrained_bert --upgrade

dic = [
    'background_image',
    'background_image_original',
    'date_uploaded',
    'date_uploaded_unix',
    'description_full',
    'genres',
    'id',
    'imdb_code',
    'language',
    'large_cover_image',
    'medium_cover_image',
    'mpa_rating',
    'rating',
    'runtime',
    'slug',
    'small_cover_image',
    'state',
    'summary',
    'synopsis',
    'title',
    'title_english',
    'title_long',
    'torrents_date_uplaoded',
    'torrents_date_uploaded_unix',
    'torrents_hash',
    'torrents_peers',
    'torrents_quality',
    'torrents_seeds',
    'torrents_size',
    'torrents_size_bytes',
    'torrents_type',
    'torrents_url',
    'url',
    'year',
    'yt_trailer_code']

a = []
b = []
c = []
d = []
e = []
f = []
g = []
h = []
i = []
j = []
k = []
l = []
m = []
n = []
o = []
p = []
q = []
r = []
s = []
t = []
u = []
v = []
w = []
x = []
y = []
z = []
aa = []
bb = []
cc = []
dd = []
ee = []
ff = []
gg = []
hh = []
ii = []

import time
start_time = []
for zz in range(1201):
  print(zz)
  url = "https://yts.mx/api/v2/list_movies.json?page=" + f'{zz+1}'
  dataset_File = cached_path(url)
  with open(dataset_File, mode = "r", encoding = "utf-8") as F:
    data = json.loads(F.read())
    movies = data['data']['movies']
    for xx in range(len(movies)):
      datum = movies[xx]
      a.append(datum[dic[0]])
      b.append(datum[dic[1]])
      if 'date_uploaded' not in datum.keys():
        c.append('none')
      else:
        c.append(datum[dic[2]])

      if 'date_uploaded_unix' not in datum.keys():
        d.append('none')
      
      else:
        d.append(datum[dic[3]])
      e.append(datum[dic[4]])
      if 'genres' not in datum.keys():
        f.append('none')
      else:
        f.append(datum[dic[5]])
      g.append(datum[dic[6]])
      h.append(datum[dic[7]])
      i.append(datum[dic[8]])
      j.append(datum[dic[9]])
      k.append(datum[dic[10]])
      l.append(datum[dic[11]])
      m.append(datum[dic[12]])
      n.append(datum[dic[13]])
      o.append(datum[dic[14]])
      p.append(datum[dic[15]])
      q.append(datum[dic[16]])
      r.append(datum[dic[17]])
      s.append(datum[dic[18]])
      t.append(datum[dic[19]])
      u.append(datum[dic[20]])
      v.append(datum[dic[21]])
      if 'torrents' not in datum.keys():
        w.append('none')
        x.append('none')
        y.append('none')
        z.append('none')
        aa.append('none')
        bb.append('none')
        cc.append('none')
        dd.append('none')
        ee.append('none')
        ff.append('none')
      else:
        w.append(datum['torrents'][0]['date_uploaded'])
        x.append(datum['torrents'][0]['date_uploaded_unix'])
        y.append(datum['torrents'][0]['hash'])
        z.append(datum['torrents'][0]['peers'])
        aa.append(datum['torrents'][0]['quality'])
        bb.append(datum['torrents'][0]['seeds'])
        cc.append(datum['torrents'][0]['size'])
        dd.append(datum['torrents'][0]['size_bytes'])
        ee.append(datum['torrents'][0]['type'])
        ff.append(datum['torrents'][0]['url'])
      gg.append(datum[dic[32]])
      hh.append(datum[dic[33]])
      ii.append(datum[dic[34]])  

data_frame = {}
lists = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, 
         aa, bb, cc, dd, ee ,ff, gg, hh, ii]
for idx in range(35):
  data_frame[dic[idx]] = lists[idx]

import pandas as pd
data_frame = pd.DataFrame(data_frame)
