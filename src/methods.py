import time
import traceback
from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from tqdm import tqdm

from src.configs import Config, SoftRuleConfig, SySurvConfig
from src.datasets import SurvivalDataset
from src.softrules import AndFinder


class Method:
    def __init__(self, data: SurvivalDataset, config: Config, n_subgroups: int):
        self.name = type(self).__name__
        self.data = data
        self.finished = False
        self.runtime = 0
        self.contrastive = None
        self.features = data.features
        self.feature_names = data.feature_names
        self.time = data.time
        self.event = data.event
        self.config = config
        self.n_subgroups = n_subgroups
        self.subgroups = []
        self.rules = []
        self.best_losses = []
        self._preprocess_data()

    def __call__(self):
        output = {'name': self.name,
                  'contrastive': self.contrastive,
                  'data': self.data(),
                  'subgroups to find': self.n_subgroups,
                  'subgroups': self.subgroups,
                  'rules': self.rules,
                  'finished': self.finished,
                  'runtime': self.runtime,
                  'config': vars(self.config)}

        return output

    def run(self, verbose=False, progress=True, prune=False):
        start = time.time()
        end = time.time()
        self.runtime = end - start
        self.finished = True

    def _preprocess_data(self):
        ...


class SoftRuleMethod(Method):
    def __init__(self, data: SurvivalDataset, config: SoftRuleConfig, n_subgroups: int):
        super().__init__(data, config, n_subgroups)
        self.scaler_x = None

    def __call__(self):
        output = super().__call__()
        
        pop_model_str = 'None'
        if hasattr(self, "population_model") and self.population_model is not None and callable(self.population_model):
            pop_model_str = self.population_model()
        output['population_model'] = pop_model_str
            
        sub_model_str = 'None'
        if hasattr(self, "subgroup_model") and self.subgroup_model is not None and callable(self.subgroup_model):
            sub_model_str = self.subgroup_model()
        output['subgroup_model'] = sub_model_str
            
        if hasattr(self, "best_losses"):
            output['best_losses'] = self.best_losses
            
        if hasattr(self, "search_time"):
            output['search_time'] = self.search_time
            
        return output

    def run(self, verbose=False, progress=True, prune=False):

        start = time.time()

        self.scaler_x = StandardScaler()
        X = self.scaler_x.fit_transform(self.features)

        X_tensor = torch.tensor(X, dtype=torch.float64)
        t_tensor = torch.tensor(self.time, dtype=torch.float64)
        e_tensor = torch.tensor(self.event, dtype=torch.float64)

        cut_points = self._init_cut_points(X_tensor)

        priors = []
        self._init_population_model()
        self.contrastive = True if self.population_model is None else False  # was swapped, now correct

        for n in range(self.n_subgroups):
            if self.n_subgroups > 1: print(f"Discovering Subgroup #{n + 1}")

            classifier = AndFinder(cut_points,
                                   temperature=self.config.temperature,
                                   use_weights=self.config.use_weights,
                                   bin_deviation=self.config.bin_deviation)

            classifier, priors = self._fit(X_tensor, t_tensor, e_tensor,
                                           classifier, priors, verbose, progress)


            classifier = classifier.cpu()
            subgroup = torch.argmax(classifier(X_tensor), dim=1).detach().numpy() == 1
            self.subgroups.append(subgroup)
            self.rules.append(classifier.get_rules(cut_points,
                                                   scaler=self.scaler_x,
                                                   feature_names=self.feature_names,
                                                   X=X))

            if prune:
                with torch.no_grad():  # for correlated datasets like the case study
                    thresh = 0.95
                    cls_cpy = deepcopy(classifier)

                    while True:
                        current_subgroup = torch.argmax(cls_cpy(X_tensor), dim=1).detach().numpy() == 1
                        if jaccard_score(subgroup, current_subgroup) < thresh:
                            break
                        
                        active_indices = [i for i, w in enumerate(cls_cpy.and_weights.relu().numpy().tolist()) if w > 0.1]
                        best_score = 0
                        best_idx = None
                        
                        for idx in active_indices:
                            temp_cls = deepcopy(cls_cpy)
                            temp_cls.and_weights[idx] = 0.0

                            score = jaccard_score(subgroup, torch.argmax(temp_cls(X_tensor), dim=1).detach().numpy() == 1)

                            if score > best_score:
                                best_score = score
                                best_idx = idx

                        if best_idx is None or best_score < thresh: 
                            break
                            
                        cls_cpy.and_weights[best_idx] = 0.0
                        if verbose:
                            print(f"pruning {self.data.feature_names[best_idx]}...")

                    print(f"\nOriginal rule:\n{self.rules[-1]}\nPruned rule:")
                    print(cls_cpy.get_rules(cut_points, scaler=self.scaler_x, feature_names=self.feature_names, X=X))
                    print(f"Original subgroup size: {subgroup.sum()}")
                        
                    pruned_subgroup = torch.argmax(cls_cpy(X_tensor), dim=1).detach().numpy() == 1
                    
                    print(f"Pruned subgroup size: {pruned_subgroup.sum()}")
                        
                    if hasattr(self, 'best_exceptionalty') and self.best_exceptionalty is not None:
                        print(f"Original subgroup exceptionality: {self.best_exceptionalty.detach()}")
                        
                        self._compute_objective(cls_cpy(X_tensor).detach(), None, None)
                        
                        print(f"Pruned subgroup exceptionality: {self.best_exceptionalty.detach()}\n")
                            
                        self.best_exceptionalities[-1] = self.best_exceptionalty.detach()
                        self.best_weighted_exceptionalities[-1] = self.best_weighted_exceptionality.detach()

                    self.rules[-1] = cls_cpy.get_rules(cut_points, scaler=self.scaler_x, feature_names=self.feature_names, X=X)
                    self.subgroups[-1] = pruned_subgroup

        if len(priors) > 0: self.subgroup_model = priors[-1]

        end = time.time()
        self.runtime = end - start
        self.finished = True

        return self.subgroups, self.rules

    def _fit(self, X, t, e, classifier, priors, verbose=False, progress=True):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        device = torch.device("cpu")
        if torch.cuda.is_available(): device = torch.device("cuda")
        batchsize = X.shape[0] if self.config.classifier_batch_size == -1 else self.config.classifier_batch_size
        e_counter = 0
        m_counter = 0

        X = X.to(device)
        t = t.to(device)
        e = e.to(device)
        classifier.to(device)
        best_classifier = classifier.state_dict()
        update_best = False

        limits = torch.stack([torch.min(X, dim=0)[0], torch.max(X, dim=0)[0]], dim=1)

        subgroup_model = self._init_subgroup_model()
        complement_model = self.population_model if not self.contrastive else self._init_subgroup_model()

        optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=self.config.classifier_lr)

        best_loss = torch.inf
        loss = best_loss

        pbar = tqdm(range(self.config.classifier_epochs), unit="epoch", mininterval=1, disable=not progress,
                    desc=f"dataset: {self.data.name}, subgroup # {len(priors) + 1}, alpha: {self.config.alpha}, lr: {self.config.classifier_lr}")
        for step in pbar:
            idx = torch.randperm(X.shape[0], device=device)
            try:
                for i in range(0, X.shape[0], batchsize):
                    idx_batch = idx[i:i + batchsize]

                    mask = torch.argmax(classifier(X[idx_batch]), dim=1).detach() == 1

                    if e_counter < self.config.e_steps:
                        self._e_step(X, t, e, idx_batch, classifier, subgroup_model, complement_model)
                        e_counter += 1

                    else:
                        loss = self._m_step(X, idx_batch, classifier, optimizer_classifier, subgroup_model, complement_model, priors)
                        if loss < best_loss:
                            best_loss = loss
                            update_best = True
                        m_counter += 1

                    pbar.set_postfix(
                        {"Loss": float(loss.detach()),
                         "Ns": float(mask.sum())})

                    if m_counter == self.config.m_steps: e_counter, m_counter = 0, 0

                if self.config.optimize_temperature:
                    if step == self.config.classifier_epochs // 2:
                        classifier.temperature = classifier.temperature / 2
                    elif step == 3 * self.config.classifier_epochs // 4:
                        classifier.temperature = classifier.temperature / 2

                if step % 100 == 0 and step > 0 and verbose:
                    self._log_step(step, classifier, X, limits)

                if update_best:
                    best_classifier = classifier.state_dict()
                    update_best = False
            except Exception:
                print(traceback.format_exc())
                break
        
        pbar.close()
        self.search_time = pbar.format_dict['elapsed']

        classifier.load_state_dict(best_classifier)

        if self.config.optimize_temperature:
            temp_grid = [2 ** (-i) for i in range(1, 11)]
            with torch.no_grad():
                best_loss = None
                for temp in temp_grid:
                    classifier.temperature = temp
                    classlabel = classifier(X)
                    loss = self._compute_objective(classlabel, subgroup_model, complement_model)
                    loss = loss.mean()
                    if best_loss is None or loss < best_loss: best_loss, best_temp = loss, temp
                classifier.temperature = best_temp

        self.best_losses.append(best_loss.detach())

        if subgroup_model is not None: priors.append(subgroup_model)

        return classifier, priors

    def _init_cut_points(self, X_tensor):
        cut_points = torch.zeros((self.features.shape[1], 2))
        for i in range(X_tensor.shape[1]):
            cut_points[i, 0] = torch.quantile(X_tensor[:, i], 0)
            cut_points[i, 1] = torch.quantile(X_tensor[:, i], 1)
        cut_points = torch.sort(cut_points, dim=1)[0]
        return cut_points

    def _init_population_model(self):
        self.population_model = None

    def _init_subgroup_model(self):
        return None

    def _e_step(self, X, t, e, idx, classifier, subgroup_model, complement_model):
        return

    def _m_step(self, X, idx, classifier, optimizer_classifier, subgroup_model, complement_model, priors=None):
        optimizer_classifier.zero_grad()
        classlabel = classifier(X[idx])
        loss = self._compute_objective(classlabel, subgroup_model, complement_model, idx, priors)
        loss = loss.sum()
        loss.backward()
        optimizer_classifier.step()

        classifier.fix_parameters()

        return loss

    def _compute_objective(self, classlabel, subgroup_model, complement_model, idx=None, priors=None):
        return self._account_for_subgroup_size(classlabel) * (
                    classlabel + self._account_for_priors(subgroup_model, priors))

    def _log_step(self, step, classifier, X, limits):
        print(f'step {step}, rule: {classifier.get_rules(limits, self.feature_names, self.scaler_x, X)}')

    def _account_for_subgroup_size(self, classlabel):
        return classlabel[:, 1].float().reshape(-1, 1).sum(axis=0) ** self.config.alpha

    def _account_for_priors(self, classlabel, subgroup_model, priors):
        return 0


class SySurv(SoftRuleMethod):
    def __init__(self, data: SurvivalDataset, config: SySurvConfig, n_subgroups: int):
        super().__init__(data, config, n_subgroups)
        self.config.contrastive = False
        self.subgroup_model = None
        self.best_exceptionalty = None
        self.best_weighted_exceptionality = None
        self.best_exceptionalities = []
        self.best_weighted_exceptionalities = []

    def _fit(self, X, t, e, classifier, priors, verbose=False, progress=True):
        res1, res2 = super()._fit(X, t, e, classifier, priors, verbose, progress)
        self.best_weighted_exceptionalities.append(self.best_weighted_exceptionality.detach())
        self.best_exceptionalities.append(self.best_exceptionalty.detach())
        return res1, res2

    def _init_cut_points(self, X_tensor):
        if not self.config.contrastive:
            return super()._init_cut_points(X_tensor)

        cut_points = torch.zeros((self.features.shape[1], 2))
        cut_points[0, 0] = torch.quantile(X_tensor[:, 0], 0)
        cut_points[0, 1] = torch.quantile(X_tensor[:, 0], 1 - 0.2)
        for i in range(1, X_tensor.shape[1]):
            cut_points[i, 0] = torch.quantile(X_tensor[:, i], 0)
            cut_points[i, 1] = torch.quantile(X_tensor[:, i], 1)
        cut_points = torch.sort(cut_points, dim=1)[0]
        return cut_points

    def _init_population_model(self):
        x = self.data.features
        y = Surv.from_arrays(self.data.event, self.data.time)
        
        if hasattr(self, 'population_model'):
            if not hasattr(self, "preds"):
                s = self.population_model.predict_survival_function(x, return_array=True)
                self.preds = torch.Tensor(s)
            return

        # TODO: add to config
        self.population_model = None if self.config.contrastive else RandomSurvivalForest(
            n_estimators=300, 
            max_depth=self.data.n_features*2, 
            random_state=self.config.seed,
            verbose=0,
            max_samples=2000/self.data.n_samples if self.data.n_samples > 2000 else self.data.n_samples,
            min_samples_split=40, 
            min_samples_leaf=20, 
            max_features=None, 
            n_jobs=-1,
        )

        if hasattr(self, 'preds'): return

        if self.population_model is not None:
            self.population_model.fit(x, y)
            s = self.population_model.predict_survival_function(x, return_array=True)
            self.preds = torch.Tensor(s)


    def _init_subgroup_model(self):
        return self.population_model

    def _e_step(self, X, t, e, idx, classifier, subgroup_model, complement_model):
        ...

    def _compute_objective(self, classlabel, subgroup_model, complement_model, idx=None, priors=None):
        membership = classlabel[:, 1].float().reshape(-1, 1)
        ns = membership.sum(axis=0)

        if not hasattr(self, 'const'):
            p_sg = self.preds
            p_comp = self.preds.mean(axis=0)
            diff = p_sg - p_comp
            self.dx = torch.diff(torch.Tensor(self.population_model.unique_times_)).unsqueeze(0)
            abs_diff = diff.abs()
            self.const = [(0.5 * self.dx * (abs_diff[:, :-1] + abs_diff[:, 1:])).sum(axis=1)]

        if idx is None: idx = np.ones_like(membership.squeeze(), dtype=bool)

        if len(self.subgroups) == 0:
            L1 = membership.squeeze() * self.const[0][idx]
            adj_L1 = (L1.sum(axis=0) * ns ** (self.config.alpha - 1))
        else:
            diversity = 0

            for index, subgroup in enumerate(self.subgroups):
                if index+1 == len(self.const):
                    diff = self.preds - self.preds[subgroup].mean(axis=0)
                    abs_diff = diff.abs()
                    self.const.append((0.5 * self.dx * (abs_diff[:, :-1] + abs_diff[:, 1:])).sum(axis=1))

                diversity += (membership.squeeze() * self.const[index+1][idx]).sum(axis=0)

            adj_L1 = ((membership.squeeze() * self.const[0][idx]).sum(axis=0) + diversity) * ns ** (self.config.alpha - 1) 
            
        self.best_weighted_exceptionality = (membership.squeeze() * self.const[0][idx]).sum(axis=0) * ns ** (self.config.alpha - 1)
        self.best_exceptionalty = (membership.squeeze() * self.const[0][idx]).sum(axis=0) / ns

        return -adj_L1

    def _account_for_priors(self, preds, classlabel, subgroup_model, priors):
        res = 0

        if not hasattr(self, 'prior_means'): self.prior_means = []

        membership = classlabel[:, 1].float().reshape(-1, 1)
        ns = membership.sum(axis=0)

        # why regularize this and not the population, arguably one would want the subgroups to differ from each other
        # rather than from the overall (in the anomaly approach, that is)

        for idx, subgroup in enumerate(self.subgroups):
            if idx == len(self.prior_means): self.prior_means.append(preds[subgroup].mean(axis=0))
            curr_mean = self.prior_means[idx]

            res += (membership.squeeze() * (preds - curr_mean).abs().sum(axis=1)).sum(axis=0) / ns

        return ns ** self.config.alpha * self.config.lambd * res