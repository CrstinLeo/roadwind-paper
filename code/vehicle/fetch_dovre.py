from remotezip import RemoteZip
url="https://zenodo.org/api/records/18483903/files/NOR-XWIND.zip/content"
with RemoteZip(url) as z:
    for i in z.infolist():
        f=i.filename
        if ("day_201029/01_serial" in f or "day_201031/01_serial" in f) and f.endswith(".vbo"):
            z.extract(i, "raw"); print("ok", f, flush=True)
