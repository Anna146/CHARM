from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

with open(project_dir + "/data/raw_data/texts_hobby.txt", "w") as out_hob, open(project_dir + "/data/raw_data/texts_profession.txt", "w") as out_prof:
    for line in open(project_dir + "/data/raw_data/texts.txt"):
        if line.split(",")[0] == "hobby":
            out_hob.write(",".join(line.split(",")[1:]))
        else:
            out_prof.write(",".join(line.split(",")[1:]))