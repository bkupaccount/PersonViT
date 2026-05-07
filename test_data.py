from transreid_pytorch.datasets import make_dataloader
from transreid_pytorch.config import cfg

cfg.defrost()

cfg.merge_from_file("/home/itit/MaulDir/projects/ppeds/research_notebook/PersonViT/transreid_pytorch/configs/cmsmt17/vit_base_ics_384.yml")

# cfg.DATASETS.NAMES = 'cmarket1501'
cfg.DATASETS.ROOT_DIR = "/home/itit/MaulDir/projects/ppeds/research_notebook/PersonViT/datasets"
cfg.freeze()

print(cfg)

data = make_dataloader(cfg)

print(data)

