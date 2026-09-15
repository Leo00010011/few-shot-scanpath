import argparse
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from os.path import join
import json
from PIL import Image
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import scipy.ndimage as filters
from torchvision.transforms import transforms
from tqdm import tqdm
from scipy.io import loadmat
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    # F5 builds the feature path by CONCATENATION, exp_key + ".pth" -- never the
    # upstream `img_name.replace('jpg', 'pth')`, which is unanchored and would corrupt
    # any key containing that substring (FR2.5, TechStack section 3.2). Reusing
    # exp_key_filename() re-asserts the [A-Za-z0-9_]+ and no-jpg rules at the call
    # site instead of assuming them.
    from eve_prep.trial_keys import exp_key_filename
except ImportError as _exc:  # pragma: no cover - a PYTHONPATH problem, not a bug here
    raise ImportError(
        "the EVE branch needs tools/ on PYTHONPATH for "
        "eve_prep.trial_keys.exp_key_filename (bash/test_eve.sh exports it): "
        "{}".format(_exc))


class EVE(Dataset):
    """
    get EVE data
    """

    def __init__(self,
                 args,
                 stimuli_dir,
                 feature_dir,
                 fixations_dir,
                 task_emb_dir,
                 action_map=(30, 40),
                 origin_size=(1680, 1050),
                 resize=(240, 320),
                 max_length=16,
                 blur_sigma=1,
                 type="train",
                 transform=None):
        self.args = args
        self.stimuli_dir = stimuli_dir
        self.feature_dir = feature_dir
        self.fixations_dir = fixations_dir
        self.action_map = action_map
        self.origin_size = origin_size
        self.resize = resize
        self.max_length = max_length
        self.blur_sigma = blur_sigma
        self.type = type
        self.transform = transform
        self.PAD = [-3, -3, -3]

        self.downscale_x = origin_size[1] / action_map[1]
        self.downscale_y = origin_size[0] / action_map[0]

        # target embeddings
        self.embedding_dict = np.load(task_emb_dir,
                                      allow_pickle=True).item()

        self.fixations_file = self.fixations_dir
        with open(self.fixations_file) as json_file:
            fixations = json.load(json_file)
        fixations = [_ for _ in fixations if _["split"] == type]

        if self.args.ex_subject[0] != -1:
            fixations = adjust_subjects(fixations, self.args.ex_subject)
        elif self.args.fewshot_subject[0] != -1:
            fixations = select_fewshot_subject(args.log_root, fixations, 
            self.args.fewshot_subject, self.args.num_fewshot, 
            self.args.random_support, self.args.subject_num, self.type)

        self.fixations = fixations
        
        # for sp in self.fixations:
        #     print(sp['name'])
        #     print(sp['subject'])
        #     print(sp['X'])

        print('total training set: ', len(self.fixations))

        self.imgid_to_sub = {}
        for index, fixation in enumerate(self.fixations):
            self.imgid_to_sub.setdefault(fixation['name'], []).append(index)
        self.imgid = list(self.imgid_to_sub.keys())


    def __len__(self):
        return len(self.imgid)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        img_name = self.imgid[idx]
        img_path = join(self.feature_dir, img_name.replace('jpg', 'pth'))
        image_ftrs = torch.load(img_path).unsqueeze(0)

        images = []
        subjects = []
        tasks = []
        task_embeddings = []
        target_scanpaths = []
        durations = []
        action_masks = []
        duration_masks = []
        for ids in self.imgid_to_sub[img_name]:
            fixation = self.fixations[ids]

            scanpath = np.zeros((self.max_length, self.action_map[0], self.action_map[1]), dtype=np.float32)
            # the first element denotes the termination action
            target_scanpath = np.zeros((self.max_length, self.action_map[0] * self.action_map[1] + 1), dtype=np.float32)
            duration = np.zeros(self.max_length, dtype=np.float32)
            action_mask = np.zeros(self.max_length, dtype=np.float32)
            duration_mask = np.zeros(self.max_length, dtype=np.float32)
            task = "free-viewing"

            pos_x = np.array(fixation["X"]).astype(np.float32)
            pos_y = np.array(fixation["Y"]).astype(np.float32)
            duration_raw = np.array(fixation["T"]).astype(np.float32)

            pos_x_discrete = np.zeros(self.max_length, dtype=np.int32) - 1
            pos_y_discrete = np.zeros(self.max_length, dtype=np.int32) - 1
            for index in range(len(pos_x)):
                # only preserve the max_length ground-truth
                if index == self.max_length:
                    break
                # since pixel is start from 1 ~ max based on matlab
                pos_x_discrete[index] = ((pos_x[index] - 1) / self.downscale_x).astype(np.int32)
                pos_y_discrete[index] = ((pos_y[index] - 1) / self.downscale_y).astype(np.int32)
                duration[index] = duration_raw[index] / 1000.0
                action_mask[index] = 1
                duration_mask[index] = 1
            if action_mask.sum() <= self.max_length - 1:
                action_mask[int(action_mask.sum())] = 1

            for index in range(self.max_length):
                if pos_x_discrete[index] == -1 or pos_y_discrete[index] == -1:
                    target_scanpath[index, 0] = 1
                else:
                    scanpath[index, pos_y_discrete[index], pos_x_discrete[index]] = 1
                    if self.blur_sigma:
                        scanpath[index] = filters.gaussian_filter(scanpath[index], self.blur_sigma)
                        scanpath[index] /= scanpath[index].sum()
                    target_scanpath[index, 1:] = scanpath[index].reshape(-1)

            images.append(image_ftrs)
            durations.append(duration)
            action_masks.append(action_mask)
            duration_masks.append(duration_mask)
            subjects.append(fixation["subject"])
            tasks.append(task)
            task_embedding = self.embedding_dict[task]
            task_embeddings.append(task_embedding)
            target_scanpaths.append(target_scanpath)


        images = torch.cat(images)
        subjects = np.array(subjects)
        task_embeddings = np.array(task_embeddings)
        target_scanpaths = np.array(target_scanpaths)
        durations = np.array(durations)
        action_masks = np.array(action_masks)
        duration_masks = np.array(duration_masks)

        # self.show_image(image/255)
        # self.show_image(image_resized/255)

        return {
            "image": images,
            "subject": subjects,
            "img_name": img_name,
            "duration": durations,
            "action_mask": action_masks,
            "duration_mask": duration_masks,
            "task": tasks,
            "task_embedding": task_embeddings,
            "target_scanpath": target_scanpaths
        }

    def collate_func(self, batch):

        img_batch = []
        subject_batch = []
        img_name_batch = []
        duration_batch = []
        action_mask_batch = []
        duration_mask_batch = []
        task_batch = []
        task_embedding_batch = []
        target_scanpath_batch = []

        for sample in batch:
            tmp_img, tmp_subject, tmp_img_name, tmp_duration, tmp_action_mask, tmp_duration_mask, \
                tmp_task, tmp_task_embedding, tmp_target_scanpath =\
                sample["image"], sample["subject"], sample["img_name"],\
                sample["duration"], sample["action_mask"], sample["duration_mask"], sample["task"], sample["task_embedding"], sample["target_scanpath"]
            img_batch.append(tmp_img)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            duration_batch.append(tmp_duration)
            action_mask_batch.append(tmp_action_mask)
            duration_mask_batch.append(tmp_duration_mask)
            task_batch.append(tmp_task)
            task_embedding_batch.append(tmp_task_embedding)
            target_scanpath_batch.append(tmp_target_scanpath)

        data = dict()
        data["images"] = torch.cat(img_batch)
        data["subjects"] = np.concatenate(subject_batch)
        data["img_names"] = img_name_batch
        data["durations"] = np.concatenate(duration_batch)
        data["action_masks"] = np.concatenate(action_mask_batch)
        data["duration_masks"] = np.concatenate(duration_mask_batch)
        data["tasks"] = task_batch
        data["task_embeddings"] = np.concatenate(task_embedding_batch)
        data["target_scanpaths"] = np.concatenate(target_scanpath_batch)

        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor
        data = {k: v.unsqueeze(0) if type(v) is torch.Tensor else v for k, v in data.items()}
        return data


class EVE_rl(Dataset):
    """
    get EVE data for evaluation
    """

    def __init__(self,
                 args,
                 stimuli_dir,
                 feature_dir,
                 fixations_dir,
                 task_emb_dir,
                 action_map=(30, 40),
                 origin_size=(600, 800),
                 resize=(240, 320),
                 type="validation",
                 transform=None):
        self.args = args
        self.stimuli_dir = stimuli_dir
        self.feature_dir = feature_dir
        self.fixations_dir = fixations_dir
        self.action_map = action_map
        self.origin_size = origin_size
        self.resize = resize
        self.type = type
        self.transform = transform

        self.downscale_x = origin_size[1] / action_map[1]
        self.downscale_y = origin_size[0] / action_map[0]

        self.resizescale_x = origin_size[1] / resize[1]
        self.resizescale_y = origin_size[0] / resize[0]

        # target embeddings
        self.embedding_dict = np.load(task_emb_dir,
                                      allow_pickle=True).item()

        self.fixations_file = self.fixations_dir
        with open(self.fixations_file) as json_file:
            fixations = json.load(json_file)
        fixations = [_ for _ in fixations if _["split"] == type]

        if self.args.ex_subject[0] != -1:
            fixations = adjust_subjects(fixations, self.args.ex_subject)
        elif self.args.fewshot_subject[0] != -1:
            with open(
                'result/{}/sampling/{}_shot/sample_{}.json'.format(
                    args.log_root.split('/')[-1], self.args.num_fewshot, self.args.random_support)) as f:
                fixations = json.load(f)
        self.fixations = fixations

        self.fixations = fixations

        self.imgid_to_sub = {}
        for index, fixation in enumerate(self.fixations):
            self.imgid_to_sub.setdefault(fixation['name'], []).append(index)
        self.imgid = list(self.imgid_to_sub.keys())

    def __len__(self):
        # return len(self.imgid) * 15
        # return len(self.fixations)
        return len(self.imgid)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        img_name = self.imgid[idx]
        img_path = join(self.feature_dir, img_name.replace('jpg', 'pth'))
        image_ftrs = torch.load(img_path).unsqueeze(0)

        # image = Image.open(img_path).convert('RGB')
        # if self.transform is not None:
        #     image = self.transform(image)

        images = []
        fix_vectors = []
        subjects = []
        firstfixs = []
        tasks = []
        task_embeddings = []
        for ids in self.imgid_to_sub[img_name]:
            fixation = self.fixations[ids]
            task = "free-viewing"

            # normalize
            x_start = np.array(fixation["X"]).astype(np.float32) / self.resizescale_x
            y_start = np.array(fixation["Y"]).astype(np.float32) / self.resizescale_y
            duration = np.array(fixation["T"]).astype(np.float32) / 1000.0

            length = fixation["length"]

            # in the middle of the image
            firstfix = np.array([self.resize[0] / 2, self.resize[1] / 2], np.int64)

            fix_vector = []
            for order in range(length):
                fix_vector.append((x_start[order], y_start[order], duration[order]))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})

            task_embedding = self.embedding_dict[task]

            fix_vectors.append(fix_vector)
            subjects.append(fixation["subject"])
            firstfixs.append(firstfix)
            images.append(image_ftrs)
            tasks.append(task)
            task_embeddings.append(task_embedding)

        images = torch.cat(images)
        return {
            "image": images,
            "fix_vectors": fix_vectors,
            "firstfix": firstfixs,
            "img_name": img_name,
            "subject": subjects,
            "task": tasks,
            "task_embedding": task_embeddings
        }

    def collate_func(self, batch):

        img_batch = []
        fix_vectors_batch = []
        firstfix_batch = []
        subject_batch = []
        img_name_batch = []
        task_batch = []
        task_embedding_batch = []

        for sample in batch:
            tmp_img, tmp_fix_vectors, tmp_firstfix, tmp_subject, tmp_img_name, tmp_task, tmp_task_embedding = \
                sample["image"], sample["fix_vectors"], sample["firstfix"], sample["subject"], sample["img_name"], sample["task"], sample[
                    "task_embedding"]
            img_batch.append(tmp_img)
            fix_vectors_batch.append(tmp_fix_vectors)
            firstfix_batch.append(tmp_firstfix)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            task_batch.append(tmp_task)
            task_embedding_batch.append(tmp_task_embedding)

        data = {}
        data["images"] = torch.cat(img_batch)
        data["fix_vectors"] = fix_vectors_batch
        data["firstfixs"] = np.concatenate(firstfix_batch)
        data["subjects"] = np.concatenate(subject_batch)
        data["img_names"] = img_name_batch
        data["tasks"] = task_batch
        data["task_embeddings"] = np.concatenate(task_embedding_batch)

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor
        data = {k: v.unsqueeze(0) if type(v) is torch.Tensor else v for k, v in data.items()}

        return data

class EVE_evaluation(Dataset):
    """
    get EVE data for evaluation
    """

    def __init__(self,
                 args,
                 stimuli_dir,
                 feature_dir,
                 fixations_dir,
                 task_emb_dir,
                 heatmaps_dir=None,
                 exp_key_map=None,
                 action_map=(24, 32),
                 origin_size=(1080, 1920),
                 resize=(384, 512),
                 max_length=16,
                 type="validation",
                 transform=None):
        self.args = args
        # FR4.2 -- stored for signature parity and NEVER read. The bridge's
        # data/eve_bridge/stimuli/ export is one .jpg per stimulus_name, which cannot
        # represent what every participant saw (OPEN-6); the per-trial screen capture
        # lives in the feature tensor instead.
        self.stimuli_dir = stimuli_dir
        self.feature_dir = feature_dir
        self.fixations_dir = fixations_dir
        self.heatmaps_dir = heatmaps_dir
        self.exp_key_map = exp_key_map
        self.action_map = action_map
        self.origin_size = origin_size
        self.resize = resize
        self.max_length = max_length
        self.type = type
        self.transform = transform

        self.downscale_x = origin_size[1] / action_map[1]
        self.downscale_y = origin_size[0] / action_map[0]

        # (1920/512, 1080/384) = (3.75, 2.8125) -- NON-UNIFORM, the deliberate squash
        # of 16:9 into the 512x384 metric screen (OPEN-4), taken so every ScanMatch and
        # MultiMatch parameter stays bit-identical to the published configuration.
        # The OSIE default origin_size=(600, 800) would give 1.5625/1.5625 here and
        # every metric would still return a plausible number (D3).
        self.resizescale_x = origin_size[1] / resize[1]
        self.resizescale_y = origin_size[0] / resize[0]

        # target embeddings
        self.embedding_dict = np.load(task_emb_dir,
                                      allow_pickle=True).item()

        self.fixations_file = self.fixations_dir
        with open(self.fixations_file) as json_file:
            fixations = json.load(json_file)
        fixations = [_ for _ in fixations if _["split"] == type]

        # FR4.4 (D4) -- stamped BEFORE the call, because select_fewshot_subject()
        # rewrites item["subject"] in place and returns a FILTERED list of the same
        # dicts. Tagging each record rather than keeping a positional list is what
        # lets the check survive filtering: a --fewshot_subject list that does not
        # cover the cohort drops records, which is a COVERAGE failure the preflight
        # owns (FR3.8's ragged-image check). This assertion is about IDENTITY, and it
        # must fire on a permuted list whether or not the count also changed.
        for _rec in fixations:
            _rec["_d4_subject_before"] = int(_rec["subject"])

        if self.args.ex_subject[0] != -1:
            fixations = adjust_subjects(fixations, self.args.ex_subject)
        elif self.args.fewshot_subject[0] != -1:
            fixations = select_fewshot_subject(None, fixations,
            self.args.fewshot_subject, self.args.num_fewshot,
            self.args.random_support, self.args.subject_num, self.type)

            # ---- FR4.4, the identity-remap assertion. THE live D4 trap -----------
            # select_fewshot_subject() remaps --fewshot_subject to a dense 0..N-1
            # range and gazeformer indexes subject_embed[subjects] with the result.
            # ONLY an ascending 0 1 2 ... N-1 argument yields the identity remap --
            # and only under the identity does exp_key_map's (name, subject) key still
            # address the right trial and row i of F3's tensor still belong to
            # participant i. Any other order silently permutes the entire cohort's
            # embeddings, and every metric still looks entirely plausible.
            for i, rec in enumerate(fixations):
                before = rec.pop("_d4_subject_before")
                if before != int(rec["subject"]):
                    raise ValueError(
                        "D4: select_fewshot_subject() remapped subject {} -> {} at "
                        "record {} ({}). --fewshot_subject must be ascending 0..N-1; "
                        "any other order silently permutes the cohort's embeddings "
                        "and every metric still looks plausible.".format(
                            before, int(rec["subject"]), i, rec["name"]))

        # The tag is scaffolding, not data: strip it from whatever survived so the
        # records stay the TechStack section 3.1 schema exactly.
        for _rec in fixations:
            _rec.pop("_d4_subject_before", None)

        self.fixations = fixations

        self.imgid_to_sub = {}
        for index, fixation in enumerate(self.fixations):
            self.imgid_to_sub.setdefault(fixation['name'], []).append(index)
        self.imgid = list(self.imgid_to_sub.keys())

    def __len__(self):
        # return len(self.imgid) * 15
        # return len(self.fixations)
        return len(self.imgid)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        img_name = self.imgid[idx]

        images = []
        fix_vectors = []
        subjects = []
        firstfixs = []
        tasks = []
        task_embeddings = []
        trial_keys = []
        lengths = []
        for ids in self.imgid_to_sub[img_name]:
            fixation = self.fixations[ids]
            task = "free-viewing"

            # ---- FR5, OPEN-6 at Stage D ------------------------------------------
            # The torch.load lives INSIDE the subject loop: EVE presents each
            # photograph at a per-trial display scale, so one tensor per stimulus name
            # cannot represent what every participant saw. Upstream filled these three
            # slots with three references to one tensor; here each gets the capture its
            # own participant actually looked at. Nothing the evaluator sees changes --
            # imgid_to_sub, __len__ over images, the uniform subject count, the
            # positional diagonal and the retrieval block are all untouched.
            subject = int(fixation["subject"])
            try:
                exp_key = self.exp_key_map[(img_name, subject)]
            except KeyError:
                raise KeyError(
                    "no exp_key for trial ({!r}, {}) -- gt_heatmaps.h5 and "
                    "fixations.json disagree (FR5.3). There is no fallback key and no "
                    "sibling subject's tensor: a mis-mapped trial hands a participant "
                    "another participant's screen and every metric still looks "
                    "plausible.".format(img_name, subject))
            image_ftrs = torch.load(join(self.feature_dir,
                                         exp_key_filename(exp_key)))
            if tuple(image_ftrs.shape) != (768, 2048) or image_ftrs.dtype != torch.float32:
                raise ValueError(
                    "feature tensor for exp_key {} is {} {}; expected (768, 2048) "
                    "torch.float32 (FR5.2)".format(
                        exp_key, tuple(image_ftrs.shape), image_ftrs.dtype))
            image_ftrs = image_ftrs.unsqueeze(0)

            # normalize
            x_start = np.array(fixation["X"]).astype(np.float32) / self.resizescale_x
            y_start = np.array(fixation["Y"]).astype(np.float32) / self.resizescale_y
            duration = np.array(fixation["T"]).astype(np.float32) / 1000.0

            length = fixation["length"]

            # in the middle of the image
            firstfix = np.array([self.resize[0] / 2, self.resize[1] / 2], np.int64)

            fix_vector = []
            for order in range(length):
                fix_vector.append((x_start[order], y_start[order], duration[order]))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})

            task_embedding = self.embedding_dict[task]

            fix_vectors.append(fix_vector)
            subjects.append(fixation["subject"])
            firstfixs.append(firstfix)
            images.append(image_ftrs)
            tasks.append(task)
            task_embeddings.append(task_embedding)
            # FR4.6 -- the two additive keys. trial_key is what GtHeatmapStore rows are
            # addressed by; length is the heatmap block's per-trial valid-step count.
            trial_keys.append("{}|{}".format(img_name, subject))
            lengths.append(min(int(length), self.max_length))

        images = torch.cat(images)
        return {
            "image": images,
            "fix_vectors": fix_vectors,
            "firstfix": firstfixs,
            "img_name": img_name,
            "subject": subjects,
            "task": tasks,
            "task_embedding": task_embeddings,
            "trial_key": trial_keys,
            "length": np.array(lengths, dtype=np.int32)
        }

    def collate_func(self, batch):

        img_batch = []
        fix_vectors_batch = []
        firstfix_batch = []
        subject_batch = []
        img_name_batch = []
        task_batch = []
        task_embedding_batch = []
        trial_key_batch = []
        length_batch = []

        for sample in batch:
            tmp_img, tmp_fix_vectors, tmp_firstfix, tmp_subject, tmp_img_name, tmp_task, tmp_task_embedding = \
                sample["image"], sample["fix_vectors"], sample["firstfix"], sample["subject"], sample["img_name"], sample["task"], sample[
                    "task_embedding"]
            img_batch.append(tmp_img)
            fix_vectors_batch.append(tmp_fix_vectors)
            firstfix_batch.append(tmp_firstfix)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            task_batch.append(tmp_task)
            task_embedding_batch.append(tmp_task_embedding)
            trial_key_batch.append(sample["trial_key"])
            length_batch.append(sample["length"])

        data = {}
        data["images"] = torch.stack(img_batch)
        data["fix_vectors"] = fix_vectors_batch
        data["firstfixs"] = np.stack(firstfix_batch)
        data["subjects"] = np.array(subject_batch)
        data["img_names"] = img_name_batch
        data["tasks"] = task_batch
        data["task_embeddings"] = np.array(task_embedding_batch)
        # (B, 3) list of str -- strings do not tensorise, and the conversion below only
        # touches ndarrays, so this passes through untouched (FR4.6).
        data["trial_keys"] = trial_key_batch
        data["lengths"] = np.stack(length_batch)

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor

        return data
    

def adjust_subjects(scanpaths, exclusive_subject):
    # Step 1: Remove elements where "subject" is in exclusive_subject
    scanpath_filtered = [item for item in scanpaths if item["subject"] not in exclusive_subject]
    # Step 2: Update the remaining "subject" values to make them continuous
    remaining_subjects = sorted(set(item["subject"] for item in scanpath_filtered))
    subject_mapping = {old: new for new, old in enumerate(remaining_subjects)}
    for item in scanpath_filtered:
        item["subject"] = subject_mapping[item["subject"]]
    return scanpath_filtered


# select num_fewshot for each subject with same group of images, but don't ensure they are selected from different task
def select_fewshot_subject(log_dir, scanpaths, fewshot_subjects, num_fewshot, k, subject_num, split):
    filtered_scanpath = [item for item in scanpaths if item["subject"] in fewshot_subjects]
    subject_mapping = {old_subject: new_subject for new_subject, old_subject in enumerate(fewshot_subjects)}
    for item in filtered_scanpath:
        item["subject"] = subject_mapping[item["subject"]]
    if split != 'train':
        return filtered_scanpath
    
    subjects = set(item['subject'] for item in filtered_scanpath)
    img_names = list(set(item['name'] for item in filtered_scanpath))
    sampled_data = []
    # Randomly select num_fewshot image names
    random.shuffle(img_names)
    selected_img_names = img_names[0:num_fewshot]
    # Group data by img_name and subject
    img_name_groups = defaultdict(lambda: defaultdict(list))
    for item in filtered_scanpath:
        img_name = item['name']
        subject = item['subject']
        img_name_groups[img_name][subject].append(item)

    # Step 4: Create the list of selected samples
    selected_samples = []
    for img_name in selected_img_names:
        for subject in subjects:
            if subject in img_name_groups[img_name]:  # Ensure the subject is available for this image
                selected_samples.extend(img_name_groups[img_name][subject])
    # for sp in selected_samples:
    #     print(sp['task'], sp['name'], sp['subject'])
    #     print(sp['X'][0:4])
    # print('=================')

    # with open('../../../PHAT/result/{}/sampling/{}_shot/sample_{}.json'.format(
    #                 args.log_root.split('/')[-1], self.args.num_fewshot, self.args.random_support)) as f:
    #     selected_samples = json.load(f)

    # print('======================')
    
    if split == 'train':
        save_sample_path = 'result/{}/sampling/{}_shot'.format(log_dir.split('/')[-1], num_fewshot)
        if not os.path.exists(save_sample_path):
            os.makedirs(save_sample_path)
        with open('{}/sample_{}.json'.format(save_sample_path, k), 'w') as f:
            json.dump(selected_samples, f, indent=4)
        return selected_samples


        