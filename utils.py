from os.path import join, isfile
from shutil import copytree, rmtree, ignore_patterns
import torch

def backup_code(src_dir, dest_dir):
    # ignore any files but files with '.py' extension
    def get_ignored(dir, filenames):
        to_exclude = ['datasets', 'models']
        ret = []
        if dir in to_exclude:
            ret.append(dir)
        for filename in filenames:
            if join(dir, filename) in to_exclude:
                ret.append(filename)
            elif isfile(join(dir, filename)) and not filename.endswith(".py") and  not filename.endswith(".sh"):
                ret.append(filename)
        # print(ret)
        return ret
    # ignore_func = lambda d, files: [f for f in files if isfile(join(d, f)) and not f.endswith('.py') and not f.endswith('.sh')]
    rmtree(dest_dir, ignore_errors=True)
    copytree(src_dir, dest_dir, ignore=ignore_patterns("datasets*", "models*", ".git*", "*.pth"))


def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat