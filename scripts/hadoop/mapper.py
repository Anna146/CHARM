#!/usr/bin/env python
import gzip
import json
import re
import sys

patterns = [
    (re.compile("#[\w]*"), " "),
    (re.compile("[\w]*@[\w]*"), " "),
    (re.compile(r"https?:\/\/.*[\r\n]*"), ""),
    (re.compile("([^\s\w']|_)+|\d|\t|\n"), " "),
    (re.compile("\s+"), " "),
    (re.compile("\.+"), "."),
]

ids = {}
#for line in gzip.open("ids_hobby.txt"):
for line in open("ids.txt"):
    pred, uid, label, postid, prob = line.strip().split("\t")
    ids[postid] = [pred, uid, label, prob]

for line in sys.stdin:
    try:
        js = json.loads(line)
    except:
        continue
    if not isinstance(js, dict):
        continue

    if "id" not in js or js["id"] not in ids:
        continue

    txt = (js.get("selftext", "") + " " + js.get("body", "") + " " + js.get("title", "")).lower()

    for pattern, substitution in patterns:
        txt = pattern.sub(substitution, txt)

    post_len = len([x for x in txt.split(" ") if len(x.strip().strip(".")) > 1])
    if post_len > 100 or post_len < 10:
        continue

    print(",".join(ids[js["id"]][:3] + [txt] + [ids[js["id"]][-1]]))
