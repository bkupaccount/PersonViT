import torch
import numpy as np
import os
from .reranking import re_ranking


def euclidean_distance(qf, gf, chunk_size=2048):
    """Memory-efficient euclidean distance. Computes in chunks to avoid OOM.
    Returns full numpy distmat — use euclidean_distance_streamed() to avoid storing it.
    """
    m = qf.shape[0]
    n = gf.shape[0]

    qq = torch.pow(qf, 2).sum(dim=1)  # (m,)
    gg = torch.pow(gf, 2).sum(dim=1)  # (n,)

    distmat = np.zeros((m, n), dtype=np.float32)
    gf_t = gf.t()

    for i in range(0, m, chunk_size):
        end = min(i + chunk_size, m)
        dist_chunk = qq[i:end].unsqueeze(1) + gg.unsqueeze(0)
        dist_chunk.addmm_(qf[i:end], gf_t, beta=1, alpha=-2)
        distmat[i:end] = dist_chunk.cpu().numpy()
        del dist_chunk

    del gf_t
    return distmat


def _euclidean_chunk(qf_chunk, gf, qq_chunk, gg, gf_t):
    """Compute euclidean distance for a single query chunk. Returns numpy array."""
    dist = qq_chunk.unsqueeze(1) + gg.unsqueeze(0)
    dist.addmm_(qf_chunk, gf_t, beta=1, alpha=-2)
    result = dist.cpu().numpy()
    del dist
    return result


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        Memory-efficient: argsort per-row instead of bulk to avoid huge indices array.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    # Only store top-max_rank indices per query (not full num_g)
    top_indices = np.zeros((num_q, max_rank), dtype=np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # sort one row at a time (avoids full m×n indices array)
        order = np.argsort(distmat[q_idx])
        top_indices[q_idx] = order[:max_rank]

        # remove gallery samples that have the same pid and camid with query
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = (g_pids[order[keep]] == q_pid).astype(np.int32)
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP, all_AP, top_indices


def _eval_rows(dist_rows, row_offset, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluate a chunk of query rows. Returns (all_cmc, all_AP, top_indices, num_valid_q)."""
    num_rows = dist_rows.shape[0]
    top_indices = np.zeros((num_rows, max_rank), dtype=np.int32)
    all_cmc = []
    all_AP = []
    num_valid_q = 0.

    for local_idx in range(num_rows):
        q_idx = row_offset + local_idx
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = np.argsort(dist_rows[local_idx])
        top_indices[local_idx] = order[:max_rank]

        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        orig_cmc = (g_pids[order[keep]] == q_pid).astype(np.int32)
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    return all_cmc, all_AP, top_indices, num_valid_q


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        cmc, mAP, qf, gf, all_AP, indices = self._compute()
        return cmc, mAP, None, self.pids, self.camids, qf, gf

    def _compute(self, chunk_size=2048):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        # Free merged feats early
        del feats

        if self.reranking:
            print('=> Enter reranking')
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
            cmc, mAP, all_AP, indices = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
            del distmat
            return cmc, mAP, qf, gf, all_AP, indices

        # --- Streamed evaluation: never store full distmat ---
        print('=> Computing DistMat with euclidean_distance (streamed)')
        m = qf.shape[0]
        n = gf.shape[0]
        max_rank = min(self.max_rank, n)

        qq = torch.pow(qf, 2).sum(dim=1)  # (m,)
        gg = torch.pow(gf, 2).sum(dim=1)  # (n,)
        gf_t = gf.t()

        all_cmc = []
        all_AP = []
        all_indices = []
        num_valid_q = 0.

        for i in range(0, m, chunk_size):
            end = min(i + chunk_size, m)

            # Compute distance for this query chunk
            dist_chunk = _euclidean_chunk(qf[i:end], gf, qq[i:end], gg, gf_t)

            # Evaluate this chunk immediately
            chunk_cmc, chunk_AP, chunk_idx, chunk_valid = _eval_rows(
                dist_chunk, i, q_pids, g_pids, q_camids, g_camids, max_rank
            )

            all_cmc.extend(chunk_cmc)
            all_AP.extend(chunk_AP)
            all_indices.append(chunk_idx)
            num_valid_q += chunk_valid

            # Discard distance chunk — this is the key memory saving
            del dist_chunk

        del gf_t, qq, gg

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)
        indices = np.concatenate(all_indices, axis=0)

        return all_cmc, mAP, qf, gf, all_AP, indices
