import sys, os


#barrier_path = '/data1/AnnoBak/TMP/Driving/Object/barrel'
barrier_path = '/data1/AnnoBak/TMP/Driving/Object/cone/'
test_list = ['2021-05-31-15-30', '2021-06-15-10-05', '2021-06-22-18-40', '2021-08-04-14-06', '2021-08-06-11-49']
txtfiles = os.listdir(barrier_path)

imgs = []
for txtfile in txtfiles:
    if txtfile[:-4] not in test_list:
        continue
    txtfpath = os.path.join(barrier_path, txtfile)
    with open(txtfpath, 'r') as f:
        fpaths = [fpath.strip() for fpath in f.readlines() if fpath.strip() != '']
        imgs.extend(fpaths)

print(len(imgs))
samples = [imgs[i] for i in range(0, len(imgs), 40)]
print(len(samples))

with open('cone_test_sample.txt', 'w') as fout:
    fout.write('\n'.join(samples))

from pathlib import Path
for sample in samples:
    os.system('cp -r {} cone_test/{}'.format(sample, Path(sample).name))
