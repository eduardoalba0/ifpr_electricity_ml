# %% md
# # Imports
# %%
import cudf as pd
import dask
import numpy as np
import pandas
import shap
import seaborn as sns
import numpy
import cupy
import keras as k
import tensorflow
import os
import random
import gc
import time
import tensorflow as tf
from cuml import train_test_split
from cuml.svm import svr
from cuml import RandomForestRegressor as CudaRandomForest
from cuml.metrics import mean_absolute_error
from dask import multiprocessing
from keras import Sequential
from keras.src.layers import Input, LSTM, Dense
import matplotlib.pyplot as plt
from pyswarms.single import GlobalBestPSO
from shap.plots import colors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from dask_cuda import LocalCUDACluster
from dask.distributed import Client

SEED = 100


def reset_seed(rnd_seed=SEED):
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(rnd_seed)
    numpy.random.seed(rnd_seed)
    cupy.random.seed(rnd_seed)
    tensorflow.random.set_seed(rnd_seed)

reset_seed()
dask.config.set(scheduler="threads", num_workers=10)

# #%% md
# # # Load Datasets
# #%%
# df_electricity = pd.read_csv('./dataset/electricity.csv', sep=";", decimal=".", header=0)
# df_climatic = pd.read_csv('./dataset/climatic.csv', sep=";", decimal=".", header=0)
#
# df_electricity["data"] = pd.to_datetime(df_electricity["data"], format="%d/%m/%Y")
# df_climatic["data"] = pd.to_datetime(df_climatic["data"], format="%d/%m/%Y")
#
# df_electricity.set_index("data", inplace=True)
# df_climatic.set_index("data", inplace=True)
#
#
# #%% md
# # # Pré-Processamento
# # ## Dados climáticos faltantes
# #%%
# for index, row in df_climatic[df_climatic.isnull()].to_pandas().iterrows():
#     df_mes = df_climatic[df_climatic["mes"] == df_climatic.at[index, "mes"]]
#     for col in row.index:
#         if pandas.isnull(df_climatic.at[index, col]):
#             df_mes.at[index, col] = df_mes[col].sum() / df_mes[col][df_mes[col].isnull() == False].count()
#             df_climatic.at[index, col] = df_mes.at[index, col]
# #%% md
# # ## Obtenção dos LAGS
# #%%
# for lag_col in ["consumo"]:
#     for i in range(1, 12 + 1):
#         lag_eletricity = df_electricity[lag_col].shift(i)
#         df_electricity[f'{lag_col}_LAG_' + '{:02d}'.format(i)] = lag_eletricity
# #%% md
# # ## União dos dados climáticos aos dados de consumo
# #%%
# df_electricity = pd.merge(left=df_electricity, right=df_climatic, on=["data", "mes", "ano"], how="left")
# df_electricity = df_electricity.drop("leitura", axis=1)
#
# #%% md
# # ## Criação das variáveis Dummy (mês e ano)
# #%%
# df_meses = pd.get_dummies(df_electricity["mes"].astype(int), prefix="", prefix_sep="", dtype=int).rename(
#     columns={"1": "mes_JAN", "2": "mes_FEV", "3": "mes_MAR", "4": "mes_ABR", "5": "mes_MAI", "6": "mes_JUN",
#              "7": "mes_JUL", "8": "mes_AGO", "9": "mes_SET", "10": "mes_OUT", "11": "mes_NOV", "12": "mes_DEZ"}
# )
# df_anos = pd.get_dummies(df_electricity["ano"].astype(int), prefix="", prefix_sep="", dtype=int).rename(
#     columns={"2017": "ano_2017", "2018": "ano_2018", "2019": "ano_2019", "2020": "ano_2020", "2021": "ano_2021",
#              "2022": "ano_2022", "2023": "ano_2023", "2024": "ano_2024"}
# )
# df_electricity = pd.concat([df_electricity, df_meses, df_anos], axis=1)
# df_electricity = df_electricity.drop(["mes", "ano"], axis=1)
# df_electricity = df_electricity.astype("float32").dropna()
#
# df_show = df_electricity.to_pandas()
# df_show
# #%% md
# #
# #%% md
# # # Análise de Correlações
# # ## Eletricidade
# # ### Correlação com os LAGS
# #%%
# corr_matrix = df_electricity[df_electricity.to_pandas().filter(like="consumo").columns].dropna().to_pandas().corr(
#     numeric_only=True)
# sns.heatmap(corr_matrix,
#             cmap="coolwarm",
#             center=0,
#             annot=True,
#             fmt='.0g')
# #%% md
# # ### Correlação com as variáveis climáticas
# #%%
# corr_matrix = df_electricity.drop(df_electricity.to_pandas().filter(like="_LAG_").columns,
#                                   axis=1).drop(df_electricity.to_pandas().filter(like="mes_").columns,
#                                                axis=1).drop(df_electricity.to_pandas().filter(like="ano_").columns,
#                                                             axis=1).dropna().to_pandas().corr(numeric_only=True)
# sns.heatmap(corr_matrix,
#             cmap="coolwarm",
#             center=0,
#             annot=True,
#             fmt='.1g')
# #%% md
# # # Análise dos SHAP values
# # ## Eletricidade
# # ### Random Forest
# #%%
# df_electricity_copy = df_electricity.copy().to_pandas()
#
# x_electricity = df_electricity_copy.drop("consumo", axis=1)
# y_electricity = df_electricity_copy["consumo"]
# model_rf_electr = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=SEED)
# shap.initjs()
#
# model_rf_electr.fit(x_electricity, y_electricity)
#
# explainer_rf_electr = shap.Explainer(model_rf_electr)
# shap_rf_electr = explainer_rf_electr(x_electricity)
#
# shap.plots.waterfall(shap_rf_electr[0], max_display=10)
# shap.plots.force(shap_rf_electr[0])
# shap.plots.bar(shap_rf_electr)
#
# importance_rf_electr = pandas.DataFrame(list(zip(x_electricity.columns, numpy.abs(shap_rf_electr.values).mean(0))),
#                                         columns=["feature", "rf importance"])
# importance_rf_electr = importance_rf_electr.sort_values(by=["rf importance"])
# importance_rf_electr
# #%% md
# # ### XGBoost
# #%%
# df_electricity_copy = df_electricity.copy().to_pandas()
#
# x_electricity = df_electricity_copy.drop("consumo", axis=1)
# y_electricity = df_electricity_copy["consumo"]
#
# model_xgb_electr = XGBRegressor(booster="gbtree", objective='reg:squarederror', random_state=SEED)
# shap.initjs()
#
# model_xgb_electr.fit(x_electricity, y_electricity)
#
# explainer_xgb_electr = shap.Explainer(model_xgb_electr)
# shap_xgb_electr = explainer_xgb_electr(x_electricity)
#
# shap.plots.waterfall(shap_xgb_electr[0], max_display=10)
# shap.plots.force(shap_xgb_electr[0])
# shap.plots.bar(shap_xgb_electr)
#
# importance_xgb_electr = pandas.DataFrame(list(zip(x_electricity.columns, numpy.abs(shap_xgb_electr.values).mean(0))),
#                                          columns=["feature", "xgb importance"])
# importance_xgb_electr = importance_xgb_electr.sort_values(by=["xgb importance"])
# importance_xgb_electr
# #%% md
# # ### Média entre RF e XGB
# #%%
# importance_electr = pandas.DataFrame(list(zip(x_electricity.columns, (
#         numpy.abs(shap_rf_electr.values).mean(0) + numpy.abs(shap_xgb_electr.values).mean(0)) / 2)),
#                                      columns=["feature", "Mean RF/XGB importance"])
# importance_electr = importance_electr.sort_values(by=["Mean RF/XGB importance"], ascending=False)
#
# plt.figure(figsize=(12, 6))
#
# bar_features_electr = list(importance_electr[0:9]["feature"])
# bar_features_electr.append(f"Sum of {len(importance_electr[9:])} other features")
# bar_importances_electr = list(importance_electr[0:9]["Mean RF/XGB importance"])
# bar_importances_electr.append(importance_electr[9:]["Mean RF/XGB importance"].sum())
#
# bar_features_electr = bar_features_electr[-1]
# bar_importances_electr = bar_importances_electr[-1]
#
# bars = plt.barh(bar_features_electr, bar_importances_electr, color=colors.red_rgb)
#
# for bar, importance in zip(bars, bar_importances_electr):
#     plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f'+{importance:.2f}',
#              va='center', ha='left', color=colors.red_rgb)
#
# plt.show()
# importance_electr = importance_electr.sort_values(by=["Mean RF/XGB importance"])
# importance_electr
#
# #%% md
# # # Seleção de Features
# # ## Eletricidade
# #%%
# importance_electr = importance_electr.sort_values(by=["Mean RF/XGB importance"])
#
#
# def ft_removal_rf(n_ft_removed, dataset):
#     df_selected = dataset[n_ft_removed:]["feature"]
#
#     x = df_electricity[df_selected]
#     y = df_electricity["consumo"]
#
#     rf_fs_electr = CudaRandomForest(n_bins=1, random_state=SEED)
#
#     cvs_electricity = []
#     for i_train, i_test in TimeSeriesSplit(n_splits=5, test_size=1).split(x, y):
#         x_train, x_test = x.iloc[i_train].to_cupy(), x.iloc[i_test].to_cupy()
#         y_train, y_test = y.iloc[i_train].to_cupy(), y.iloc[i_test].to_cupy()
#
#         rf_fs_electr.fit(x_train, y_train)
#         cvs_electricity.append(int(mean_absolute_error(y_test, rf_fs_electr.predict(x_test)).get()))
#
#     return int(numpy.array(cvs_electricity).mean())
#
#
# def ft_removal_xgb(n_ft_removed, dataset):
#     df_selected = dataset[n_ft_removed:]["feature"]
#
#     x = df_electricity[df_selected]
#     y = df_electricity["consumo"]
#
#     xgb_fs_electr = XGBRegressor(booster="gbtree", device="cuda", random_state=SEED)
#
#     cvs_electricity = []
#     for i_train, i_test in TimeSeriesSplit(n_splits=5, test_size=1).split(x, y):
#         x_train, x_test = x.iloc[i_train].to_cupy(), x.iloc[i_test].to_cupy()
#         y_train, y_test = y.iloc[i_train].to_cupy(), y.iloc[i_test].to_cupy()
#
#         xgb_fs_electr.fit(x_train, y_train)
#         cvs_electricity.append(int(mean_absolute_error(y_test, xgb_fs_electr.predict(x_test)).get()))
#
#     return int(numpy.array(cvs_electricity).mean())
#
#
# importance_electr = importance_electr.sort_values(by=["Mean RF/XGB importance"])
#
# ft_rm_rf_electr = dask.compute(
#     [dask.delayed(ft_removal_rf)(i, importance_electr) for i in range(importance_electr["feature"].shape[0])])[0]
# ft_rm_xgb_electr = dask.compute(
#     [dask.delayed(ft_removal_xgb)(i, importance_electr) for i in range(importance_electr["feature"].shape[0])])[0]
#
#
# #%%
# ft_rm_mean_electr = (np.array(ft_rm_rf_electr) + np.array(ft_rm_xgb_electr)) / 2
#
# plt.plot([-x for x in ft_rm_rf_electr[:40]], label="RF")
# plt.plot([-x for x in ft_rm_xgb_electr[:40]], label="XGBoost")
# plt.plot([-x for x in ft_rm_mean_electr[:40]], label="XGBoost/RF Mean")
# plt.xlabel('Number of features removed')
# plt.ylabel('Neg. Mean Absolute Error')
# plt.legend()
# plt.show()
#
# importance_electr = importance_electr.sort_values(by=["Mean RF/XGB importance"])
# min_index = np.argmin(ft_rm_mean_electr[:40])
# df_electricity_selected = df_electricity.drop(importance_electr[:min_index]["feature"], axis=1)
# df_electricity_selected.to_pandas().to_csv(f"dataset/elect_merged_selected.csv", sep=";", decimal=",")
# df_electricity_selected

# %% md
# # Load Datasets
df_electricity_selected = pd.read_csv('./dataset/elect_merged_selected.csv', sep=";", decimal=",", header=0).set_index(
    "data")
df_electricity_selected = df_electricity_selected.astype("float32")


# %% md
# # Configuração dos Otimizadores
# ## Algoritmo Genético
# ### Random Forest
# %%
class IndRF:
    def __init__(self):
        self.fitness = None
        self.seed = None
        self.estimators = 0
        self.max_depth = 0
        self.min_samples_split = 0
        self.min_samples_leaf = 0

    def create_random(self):
        self.rand_estimators()
        self.rand_depth()
        self.rand_samples_split()
        self.rand_samples_leaf()
        return self

    def rand_estimators(self):
        self.estimators = random.randint(1, 300)

    def rand_depth(self):
        self.max_depth = random.randint(1, 300)

    def rand_samples_split(self):
        self.min_samples_split = random.randint(2, 50)

    def rand_samples_leaf(self):
        self.min_samples_leaf = random.randint(1, 50)


class GARF:
    def __init__(self, dataset, n_individuals, n_generations, seed=SEED):
        reset_seed(seed)
        self.seed = seed
        self.dataset = dataset
        self.n_individuals = n_individuals
        self.n_generations = n_generations
        self.mutation_rate = 0.5
        self.fertility = 1
        self.population = {}
        self.iters = []
        self.init_pop()
        self.init_gen()

    def init_pop(self):
        futures = [dask.delayed(self.create_ind)(_) for _ in range(self.n_individuals)]
        self.population = dask.compute(futures)[0]
        self.population = sorted(self.population, key=lambda a: a.fitness)
        self.iters.append(self.population[0])

    def create_ind(self, i):
        print(f'Ind:{i}')
        ind = IndRF().create_random()
        ind = self.get_fitness(ind)
        ind.seed = self.seed
        return ind

    def init_gen(self):
        for i in range(self.n_generations):
            print(f"Iter: {i}")
            new_seed = self.seed * (2 + i)
            reset_seed(new_seed)

            loc_pop = self.population[:self.n_individuals - 1].copy()
            dask.compute(
                [dask.delayed(self.crossover)(random.choice(loc_pop), random.choice(loc_pop), new_seed) for j in
                 range(int(self.n_individuals * self.fertility))])

            self.population = sorted(self.population, key=lambda a: a.fitness)

            self.iters.append(self.population[0])

            self.save_pop_csv()
            self.save_iters_csv()

            del new_seed, loc_pop
            gc.collect()
            print(f"Best: {self.population[0].fitness}")

    def crossover(self, ind_a, ind_b, new_seed):
        ind = IndRF()
        ind.estimators = random.choice([ind_a.estimators, ind_b.estimators])
        ind.max_depth = random.choice([ind_a.max_depth, ind_b.max_depth])
        ind.min_samples_split = random.choice([ind_a.min_samples_split, ind_b.min_samples_split])
        ind.min_samples_leaf = random.choice([ind_a.min_samples_leaf, ind_b.min_samples_leaf])

        ind.seed = new_seed
        if random.uniform(0, 1) < self.mutation_rate:
            ind = self.mutation(ind)

        ind = self.get_fitness(ind)
        self.population.append(ind)
        return ind

    def mutation(self, ind):
        random.choice([
            ind.rand_estimators(),
            ind.rand_depth(),
            ind.rand_samples_split(),
            ind.rand_samples_leaf(),
        ])
        return ind

    def get_fitness(self, individual):
        search = list(filter(lambda ind:
                             ind.estimators == individual.estimators and
                             ind.max_depth == individual.max_depth and
                             ind.min_samples_split == individual.min_samples_split and
                             ind.min_samples_leaf == individual.min_samples_leaf, self.population))

        if search:
            return search[0]

        x = self.dataset.drop("consumo", axis=1)
        y = self.dataset["consumo"]

        model = CudaRandomForest(random_state=self.seed,
                                 n_estimators=individual.estimators,
                                 max_depth=individual.max_depth,
                                 min_samples_split=individual.min_samples_split,
                                 min_samples_leaf=individual.min_samples_leaf,
                                 n_streams=individual.estimators,
                                 n_bins=x.shape[0])

        cvs = []
        for i_train, i_test in TimeSeriesSplit(n_splits=5, test_size=1).split(x, y):
            x_train, x_test = x.iloc[i_train].to_cupy().get(), x.iloc[i_test].to_cupy().get()
            y_train, y_test = y.iloc[i_train].to_cupy().get(), y.iloc[i_test].to_cupy().get()

            model.fit(x_train, y_train)

            cvs.append(int(mean_absolute_error(y_test, model.predict(x_test))))
            del x_train, x_test, y_train, y_test

        individual.fitness = int(numpy.array(cvs).mean())

        del x, y, cvs, i_train, i_test, model
        gc.collect()
        return individual

    def population_dataframe(self):
        df = pd.DataFrame()
        for ind in self.population:
            df = pd.concat([df, pd.DataFrame({
                "N_estimators": ind.estimators,
                "Max_depth": ind.max_depth,
                "Min_samples_split": ind.min_samples_split,
                "Min_samples_leaf": ind.min_samples_leaf,
                "Seed": ind.seed,
                "Fitness": ind.fitness
            })])
        return df

    def iters_dataframe(self):
        df = pd.DataFrame()
        for ind in self.iters:
            df = pd.concat([df, pd.DataFrame({
                "N_estimators": ind.estimators,
                "Max_depth": ind.max_depth,
                "Min_samples_split": ind.min_samples_split,
                "Min_samples_leaf": ind.min_samples_leaf,
                "Seed": ind.seed,
                "Fitness": ind.fitness
            })])
        return df

    def save_pop_csv(self):
        pd_df = self.population_dataframe().to_pandas()
        pd_df.to_csv(f"results/GA-RF POP SEED {self.seed}.csv", sep=";", decimal=",", index=True)
        del pd_df

    def save_iters_csv(self):
        pd_df = self.iters_dataframe().to_pandas()
        pd_df.to_csv(f"results/GA-RF ITERS SEED {self.seed}.csv", sep=";", decimal=",", index=True)
        del pd_df


# %% md
# ### XGBoost
# %%
class IndXGB:
    def __init__(self):
        self.fitness = None
        self.seed = None
        self.estimators = 0
        self.max_depth = 0
        self.booster = None
        self.reg_lambda = 0
        self.reg_alpha = 0

    def create_random(self):
        self.rand_estimators()
        self.rand_depth()
        self.rand_booster()
        self.rand_lambda()
        self.rand_alpha()
        return self

    def rand_estimators(self):
        self.estimators = random.randint(1, 300)

    def rand_depth(self):
        self.max_depth = random.randint(1, 300)

    def rand_booster(self):
        self.booster = random.choice(["gbtree", "gblinear", "dart"])

    def rand_lambda(self):
        self.reg_lambda = random.uniform(0, 100)

    def rand_alpha(self):
        self.reg_alpha = random.uniform(0, 100)


class GAXGB:
    def __init__(self, dataset, n_individuals, n_generations, seed=SEED):
        reset_seed(seed)
        self.seed = seed
        self.dataset = dataset
        self.n_individuals = n_individuals
        self.n_generations = n_generations
        self.mutation_rate = 0.5
        self.fertility = 1
        self.population = []
        self.iters = []
        self.init_pop()
        self.init_gen()

    def init_pop(self):
        futures = [dask.delayed(self.create_ind)(_) for _ in range(self.n_individuals)]
        self.population = dask.compute(futures)[0]
        self.population = sorted(self.population, key=lambda a: a.fitness)
        self.iters.append(self.population[0])

    def create_ind(self, i):
        print(f'Ind:{i}')
        ind = IndXGB().create_random()
        ind = self.get_fitness(ind)
        ind.seed = self.seed
        self.population.append(ind)

    def init_gen(self):
        for i in range(self.n_generations):
            print(f"Iter: {i}")
            new_seed = self.seed * (2 + i)
            reset_seed(new_seed)

            loc_pop = self.population[:self.n_individuals - 1].copy()
            dask.compute(
                [dask.delayed(self.crossover)(random.choice(loc_pop), random.choice(loc_pop), new_seed) for j in
                 range(int(self.n_individuals * self.fertility))])

            self.population = sorted(self.population, key=lambda a: a.fitness)

            self.iters.append(self.population[0])

            self.save_pop_csv()
            self.save_iters_csv()

            del new_seed, loc_pop
            gc.collect()
            print(f"Best: {self.population[0].fitness}")

    def crossover(self, ind_a, ind_b, new_seed):
        ind = IndXGB()
        ind.estimators = random.choice([ind_a.estimators, ind_b.estimators])
        ind.max_depth = random.choice([ind_a.max_depth, ind_b.max_depth])
        ind.booster = random.choice([ind_a.booster, ind_b.booster])
        ind.reg_lambda = random.choice([ind_a.reg_lambda, ind_b.reg_lambda])
        ind.reg_alpha = random.choice([ind_a.reg_alpha, ind_b.reg_alpha])

        ind.seed = new_seed
        if random.uniform(0, 1) < self.mutation_rate:
            ind = self.mutation(ind)

        ind = self.get_fitness(ind)
        self.population.append(ind)
        return ind

    def mutation(self, ind):
        random.choice([
            ind.rand_estimators(),
            ind.rand_depth(),
            ind.rand_booster(),
            ind.rand_lambda(),
            ind.rand_alpha()
        ])
        return ind

    def get_fitness(self, individual):
        search = list(filter(lambda ind:
                             ind.estimators == individual.estimators and
                             ind.max_depth == individual.max_depth and
                             ind.booster == individual.booster and
                             ind.reg_lambda == individual.reg_lambda and
                             ind.reg_alpha == individual.reg_alpha, self.population))

        if search:
            return search[0]

        x = self.dataset.drop("consumo", axis=1)
        y = self.dataset["consumo"]

        updater = "coord_descent" if individual.booster == "gblinear" else None

        model = XGBRegressor(device="cuda", random_state=self.seed,
                             n_estimators=individual.estimators,
                             max_depth=individual.max_depth,
                             booster=individual.booster,
                             reg_lambda=individual.reg_lambda,
                             reg_alpha=individual.reg_alpha,
                             updater=updater)

        cvs = []
        for i_train, i_test in TimeSeriesSplit(n_splits=5, test_size=1).split(x, y):
            x_train, x_test = x.iloc[i_train].to_cupy().get(), x.iloc[i_test].to_cupy().get()
            y_train, y_test = y.iloc[i_train].to_cupy().get(), y.iloc[i_test].to_cupy().get()

            model.fit(x_train, y_train)

            cvs.append(int(mean_absolute_error(y_test, model.predict(x_test).get())))
            del x_train, x_test, y_train, y_test

        individual.fitness = int(numpy.array(cvs).mean())

        del x, y, cvs, i_train, i_test, model
        gc.collect()
        return individual

    def population_dataframe(self):
        df = pd.DataFrame()
        for ind in self.population:
            df = pd.concat([df, pd.DataFrame({
                "N_estimators": ind.estimators,
                "Max_depth": ind.max_depth,
                "Booster": ind.booster,
                "Lambda": ind.reg_lambda,
                "Alpha": ind.reg_alpha,
                "Seed": ind.seed,
                "Fitness": ind.fitness
            })])
        return df

    def iters_dataframe(self):
        df = pd.DataFrame()
        for ind in self.iters:
            df = pd.concat([df, pd.DataFrame({
                "N_estimators": ind.estimators,
                "Max_depth": ind.max_depth,
                "Booster": ind.booster,
                "Lambda": ind.reg_lambda,
                "Alpha": ind.reg_alpha,
                "Seed": ind.seed,
                "Fitness": ind.fitness
            })])
        return df

    def save_pop_csv(self):
        pd_df = self.population_dataframe().to_pandas()
        pd_df.to_csv(f"results/GA-XGB POP SEED {self.seed}.csv", sep=";", decimal=",", index=True)
        del pd_df

    def save_iters_csv(self):
        pd_df = self.iters_dataframe().to_pandas()
        pd_df.to_csv(f"results/GA-XGB ITERS SEED {self.seed}.csv", sep=";", decimal=",", index=True)
        del pd_df


# %% md
# ### SVR
# %%
class IndSVR:
    def __init__(self):
        self.fitness = None
        self.seed = None
        self.c = 0
        self.epsilon = 0
        self.degree = 0
        self.kernel = None
        self.gamma = None

    def create_random(self):
        self.rand_c()
        self.rand_epsilon()
        self.rand_kernel()
        return self

    def rand_c(self):
        self.c = random.uniform(0.001, 300)

    def rand_epsilon(self):
        self.epsilon = random.uniform(0.001, 300)

    def rand_kernel(self):
        self.kernel = random.choice(["poly", "rbf", "sigmoid"])


class GASVR:
    def __init__(self, dataset, n_individuals, n_generations, seed=SEED):
        reset_seed(seed)
        self.seed = seed
        self.dataset = dataset
        self.n_individuals = n_individuals
        self.n_generations = n_generations
        self.mutation_rate = 0.5
        self.fertility = 1
        self.population = []
        self.iters = []
        self.init_pop()
        self.init_gen()

    def init_pop(self):
        futures = [dask.delayed(self.create_ind)(_) for _ in range(self.n_individuals)]
        self.population = dask.compute(futures)[0]
        self.population = sorted(self.population, key=lambda a: a.fitness)
        self.iters.append(self.population[0])

    def create_ind(self, i):
        ind = IndSVR().create_random()
        ind = self.get_fitness(ind)
        ind.seed = self.seed
        self.population.append(ind)
        print(f'Ind:{i}')
        return ind

    def init_gen(self):
        for i in range(self.n_generations):
            print(f"Iter: {i}")
            new_seed = self.seed * (2 + i)
            reset_seed(new_seed)

            loc_pop = self.population[:self.n_individuals - 1].copy()
            dask.compute(
                [dask.delayed(self.crossover)(random.choice(loc_pop), random.choice(loc_pop), new_seed) for j in
                 range(int(self.n_individuals * self.fertility))])

            self.population = sorted(self.population, key=lambda a: a.fitness)

            self.iters.append(self.population[0])

            self.save_pop_csv()
            self.save_iters_csv()

            del new_seed, loc_pop
            gc.collect()
            print(f"Best: {self.population[0].fitness}")

    def crossover(self, ind_a, ind_b, new_seed):
        ind = IndSVR()
        ind.c = random.choice([ind_a.c, ind_b.c])
        ind.epsilon = random.choice([ind_a.epsilon, ind_b.epsilon])
        ind.kernel = random.choice([ind_a.kernel, ind_b.kernel])

        ind.seed = new_seed
        if random.uniform(0, 1) < self.mutation_rate:
            ind = self.mutation(ind)

        ind = self.get_fitness(ind)
        self.population.append(ind)
        return ind

    def mutation(self, ind):
        random.choice([
            ind.rand_c(),
            ind.rand_epsilon(),
            ind.rand_kernel(),
        ])
        return ind

    def get_fitness(self, individual):
        search = list(filter(lambda ind:
                             ind.c == individual.c and
                             ind.epsilon == individual.epsilon and
                             ind.degree == individual.degree and
                             ind.kernel == individual.kernel and
                             ind.gamma == individual.gamma, self.population))

        if search:
            return search[0]

        x = self.dataset.drop("consumo", axis=1)
        y = self.dataset["consumo"]

        model = svr.SVR(C=individual.c,
                        epsilon=individual.epsilon,
                        kernel=individual.kernel)

        cvs = []
        for i_train, i_test in TimeSeriesSplit(n_splits=5, test_size=1).split(x, y):
            x_train, x_test = x.iloc[i_train].to_cupy().get(), x.iloc[i_test].to_cupy().get()
            y_train, y_test = y.iloc[i_train].to_cupy().get(), y.iloc[i_test].to_cupy().get()

            model.fit(x_train, y_train)

            cvs.append(int(mean_absolute_error(y_test, model.predict(x_test).get())))
            del x_train, x_test, y_train, y_test

        individual.fitness = int(numpy.array(cvs).mean())

        del x, y, cvs, i_train, i_test, model
        gc.collect()
        return individual

    def population_dataframe(self):
        df = pd.DataFrame()
        for ind in self.population:
            df = pd.concat([df, pd.DataFrame({
                "C": ind.c,
                "Epsilon": ind.epsilon,
                "Kernel": ind.kernel,
                "Seed": ind.seed,
                "Fitness": ind.fitness
            })])
        return df

    def iters_dataframe(self):
        df = pd.DataFrame()
        for ind in self.iters:
            df = pd.concat([df, pd.DataFrame({
                "C": ind.c,
                "Epsilon": ind.epsilon,
                "Kernel": ind.kernel,
                "Seed": ind.seed,
                "Fitness": ind.fitness
            })])
        return df

    def save_pop_csv(self):
        pd_df = self.population_dataframe().to_pandas()
        pd_df.to_csv(f"results/GA-SVR POP SEED {self.seed}.csv", sep=";", decimal=",", index=True)
        del pd_df

    def save_iters_csv(self):
        pd_df = self.iters_dataframe().to_pandas()
        pd_df.to_csv(f"results/GA-SVR ITERS SEED {self.seed}.csv", sep=";", decimal=",", index=True)
        del pd_df


# %% md
# ### LSTM
# %%
class IndLSTM:
    def __init__(self):
        self.fitness = None
        self.seed = None
        self.lstm_units = 0
        self.epochs = 0
        self.batch_size = 0
        self.lstm_activation = None
        self.bias = None

    def create_random(self):
        self.rand_units()
        self.rand_epochs()
        self.rand_batch()
        self.rand_activation()
        self.rand_bias()
        return self

    def rand_units(self):
        self.lstm_units = random.randint(1, 300)

    def rand_epochs(self):
        self.epochs = random.randint(1, 100)

    def rand_batch(self):
        self.batch_size = random.randint(1, 300)

    def rand_activation(self):
        self.lstm_activation = random.choice(
            ["linear", "mish", "sigmoid", "softmax", "softplus", "softsign", "tanh", None])

    def rand_bias(self):
        self.bias = random.choice([False, True])


class GALSTM:
    def __init__(self, dataset, n_individuals, n_generations, seed=SEED):
        reset_seed(seed)
        self.seed = seed
        self.dataset = dataset
        self.n_individuals = n_individuals
        self.n_generations = n_generations
        self.mutation_rate = 0.5
        self.fertility = 1
        self.population = []
        self.iters = []
        self.init_pop()
        self.init_gen()

    def init_pop(self):
        futures = [dask.delayed(self.create_ind)(_) for _ in range(self.n_individuals)]
        self.population = dask.compute(futures)[0]
        self.population = sorted(self.population, key=lambda a: a.fitness)
        self.iters.append(self.population[0])

    def create_ind(self, i):
        print(f'Ind:{i}')
        ind = IndLSTM().create_random()
        ind = self.get_fitness(ind)
        ind.seed = self.seed
        self.population.append(ind)
        return ind

    def init_gen(self):
        for i in range(self.n_generations):
            print(f"Iter: {i}")
            new_seed = self.seed * (2 + i)
            reset_seed(new_seed)

            loc_pop = self.population[:self.n_individuals - 1].copy()
            dask.compute(
                [dask.delayed(self.crossover)(random.choice(loc_pop), random.choice(loc_pop), new_seed) for j in
                 range(int(self.n_individuals * self.fertility))])

            self.population = sorted(self.population, key=lambda a: a.fitness)

            self.iters.append(self.population[0])

            self.save_pop_csv()
            self.save_iters_csv()

            del new_seed, loc_pop
            tf.keras.backend.clear_session(True)
            print(f"Best: {self.population[0].fitness}")

    def crossover(self, ind_a, ind_b, new_seed):
        ind = IndLSTM()
        ind.lstm_units = random.choice([ind_a.lstm_units, ind_b.lstm_units])
        ind.epochs = random.choice([ind_a.epochs, ind_b.epochs])
        ind.batch_size = random.choice([ind_a.batch_size, ind_b.batch_size])
        ind.lstm_activation = random.choice([ind_a.lstm_activation, ind_b.lstm_activation])
        ind.bias = random.choice([ind_a.bias, ind_b.bias])

        ind.seed = new_seed
        if random.uniform(0, 1) < self.mutation_rate:
            ind = self.mutation(ind)

        ind = self.get_fitness(ind)
        self.population.append(ind)
        return ind

    def mutation(self, ind):
        random.choice([
            ind.rand_units(),
            ind.rand_epochs(),
            ind.rand_batch(),
            ind.rand_activation(),
            ind.rand_bias()
        ])
        return ind

    def get_fitness(self, individual):
        print(f"Units: {individual.lstm_units}" +
              f"Epochs: {individual.epochs}" +
              f"Batch Size: {individual.batch_size}" +
              f"Activation: {individual.lstm_activation}" +
              f"Bias: {individual.bias}")

        search = list(filter(lambda ind:
                             ind.lstm_units == individual.lstm_units and
                             ind.epochs == individual.epochs and
                             ind.batch_size == individual.batch_size and
                             ind.lstm_activation == individual.lstm_activation and
                             ind.bias == individual.bias, self.population))

        if search:
            return search[0]

        x = self.dataset.drop("consumo", axis=1)
        y = self.dataset["consumo"]

        model = Sequential([
            Input((x.shape[1], 1)),
            LSTM(individual.lstm_units,
                 activation=individual.lstm_activation,
                 use_bias=individual.bias,
                 seed=self.seed),
            Dense(1),
        ])
        model.compile(loss='mse')

        cvs = []
        for i_train, i_test in TimeSeriesSplit(n_splits=5, test_size=1).split(x, y):
            x_train, x_test = x.iloc[i_train].to_cupy().get(), x.iloc[i_test].to_cupy().get()
            y_train, y_test = y.iloc[i_train].to_cupy().get(), y.iloc[i_test].to_cupy().get()

            model.fit(x_train, y_train, shuffle=False, verbose=False, epochs=individual.epochs,
                      batch_size=individual.batch_size)
            cvs.append(int(mean_absolute_error(y_test, model.predict(x_test)[0])))
            del x_train, x_test, y_train, y_test

        individual.fitness = int(numpy.array(cvs).mean())

        del x, y, cvs, i_train, i_test, model
        gc.collect()
        return individual

    def population_dataframe(self):
        df = pd.DataFrame()
        for ind in self.population:
            df = pd.concat([df, pd.DataFrame({
                "Units": ind.lstm_units,
                "Epochs": ind.epochs,
                "Batch Size": ind.batch_size,
                "Activation": ind.lstm_activation,
                "Bias": ind.bias,
                "Seed": ind.seed,
                "Fitness": ind.fitness
            })])
        return df

    def iters_dataframe(self):
        df = pd.DataFrame()
        for ind in self.iters:
            df = pd.concat([df, pd.DataFrame({
                "Units": ind.lstm_units,
                "Epochs": ind.epochs,
                "Batch Size": ind.batch_size,
                "Activation": ind.lstm_activation,
                "Bias": ind.bias,
                "Seed": ind.seed,
                "Fitness": ind.fitness
            })])
        return df

    def save_pop_csv(self):
        pd_df = self.population_dataframe().to_pandas()
        pd_df.to_csv(f"results/GA-LSTM POP SEED {self.seed}.csv", sep=";", decimal=",", index=True)
        del pd_df

    def save_iters_csv(self):
        pd_df = self.iters_dataframe().to_pandas()
        pd_df.to_csv(f"results/GA-LSTM ITERS SEED {self.seed}.csv", sep=";", decimal=",", index=True)
        del pd_df


# %% md
# ## Enxame de Partículas
# ### Random Forest
# %%
class PartRF:
    def __init_(self):
        self.fitness = None
        self.seed = None
        self.estimators = 0
        self.max_depth = 0
        self.min_samples_split = 0
        self.min_samples_leaf = 0


class PSORF:
    def __init__(self, dataset, n_particles, n_iters, seed=SEED):
        reset_seed(seed)
        self.seed = seed
        self.dataset = dataset
        self.n_particles = n_particles
        self.n_iters = n_iters
        self.particles = []
        self.iters = []
        self.run()

    def run(self):
        lower_bound = [1, 1, 2, 1]
        uppper_bound = [300, 300, 50, 50]
        bounds = (lower_bound, uppper_bound)

        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = GlobalBestPSO(n_particles=self.n_particles,
                                  dimensions=4,
                                  options=options,
                                  bounds=bounds)

        optimizer.optimize(self.get_fitness, iters=self.n_iters)
        self.particles = sorted(self.particles, key=lambda a: a.fitness)

    def get_fitness(self, parts):
        parts = np.round(parts)
        fit_lst = dask.compute([dask.delayed(self.objective_function)(parts[j]) for j in range(self.n_particles)])[0]
        self.particles = sorted(self.particles, key=lambda a: a.fitness)
        self.iters.append(self.particles[0])

        self.save_parts_csv()
        self.save_iters_csv()

        return fit_lst

    def objective_function(self, particle_arr):
        new_seed = self.seed * (2 + int(len(self.particles) / self.n_particles))
        reset_seed(new_seed)
        particle = PartRF()
        particle.estimators = int(particle_arr[0])
        particle.max_depth = int(particle_arr[1])
        particle.min_samples_split = int(particle_arr[2])
        particle.min_samples_leaf = int(particle_arr[3])
        particle.seed = new_seed

        search = list(filter(lambda par:
                             par.estimators == particle.estimators and
                             par.max_depth == particle.max_depth and
                             par.min_samples_split == particle.min_samples_split and
                             par.min_samples_leaf == particle.min_samples_leaf, self.particles))

        if search:
            self.particles.append(search[0])
            return search[0].fitness

        x = self.dataset.drop("consumo", axis=1)
        y = self.dataset["consumo"]

        model = CudaRandomForest(random_state=self.seed,
                                 n_estimators=particle.estimators,
                                 max_depth=particle.max_depth,
                                 min_samples_split=particle.min_samples_split,
                                 min_samples_leaf=particle.min_samples_leaf,
                                 n_streams=particle.estimators,
                                 n_bins=x.shape[0])

        cvs = []
        for i_train, i_test in TimeSeriesSplit(n_splits=5, test_size=1).split(x, y):
            x_train, x_test = x.iloc[i_train].to_cupy().get(), x.iloc[i_test].to_cupy().get()
            y_train, y_test = y.iloc[i_train].to_cupy().get(), y.iloc[i_test].to_cupy().get()

            model.fit(x_train, y_train)

            cvs.append(int(mean_absolute_error(y_test, model.predict(x_test).get())))
            del x_train, x_test, y_train, y_test

        particle.fitness = int(numpy.array(cvs).mean())

        self.particles.append(particle)

        del new_seed, x, y, cvs, i_train, i_test, model
        gc.collect()
        return particle.fitness

    def particles_dataframe(self):
        df = pd.DataFrame()
        for part in self.particles:
            df = pd.concat([df, pd.DataFrame({
                "N_estimators": part.estimators,
                "Max_depth": part.max_depth,
                "Min_samples_split": part.min_samples_split,
                "Min_samples_leaf": part.min_samples_leaf,
                "Seed": part.seed,
                "Fitness": part.fitness
            })])
        return df

    def iters_dataframe(self):
        df = pd.DataFrame()
        for part in self.iters:
            df = pd.concat([df, pd.DataFrame({
                "N_estimators": part.estimators,
                "Max_depth": part.max_depth,
                "Min_samples_split": part.min_samples_split,
                "Min_samples_leaf": part.min_samples_leaf,
                "Seed": part.seed,
                "Fitness": part.fitness
            })])
        return df

    def save_parts_csv(self):
        pd_df = self.particles_dataframe().to_pandas()
        pd_df.to_csv(f"results/PSO-RF POP SEED {self.seed}.csv", sep=";", decimal=",", index=True)
        del pd_df

    def save_iters_csv(self):
        pd_df = self.iters_dataframe().to_pandas()
        pd_df.to_csv(f"results/PSO-RF ITERS SEED {self.seed}.csv", sep=";", decimal=",", index=True)
        del pd_df


# %% md
# ### XGBoost
# %%
class PartXGB:
    def __init_(self):
        self.fitness = None
        self.seed = None
        self.estimators = 0
        self.max_depth = 0
        self.booster = None
        self.reg_lambda = 0
        self.reg_alpha = 0


class PSOXGB:
    def __init__(self, dataset, n_particles, n_iters, seed=SEED):
        reset_seed(seed)
        self.seed = seed
        self.dataset = dataset
        self.n_particles = n_particles
        self.n_iters = n_iters
        self.particles = []
        self.iters = []
        self.BOOSTERS = ["gbtree", "gblinear", "dart"]
        self.run()

    def run(self):
        lower_bound = [1, 1, 0, 0, 0]
        uppper_bound = [300, 300, 2, 100, 100]
        bounds = (lower_bound, uppper_bound)

        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = GlobalBestPSO(n_particles=self.n_particles,
                                  dimensions=5,
                                  options=options,
                                  bounds=bounds)

        optimizer.optimize(self.get_fitness, iters=self.n_iters)
        self.particles = sorted(self.particles, key=lambda a: a.fitness)

    def get_fitness(self, parts):
        parts = np.round(parts)
        fit_lst = dask.compute([dask.delayed(self.objective_function)(parts[j]) for j in range(self.n_particles)])[0]
        self.particles = sorted(self.particles, key=lambda a: a.fitness)
        self.iters.append(self.particles[0])

        self.save_parts_csv()
        self.save_iters_csv()

        gc.collect()
        return fit_lst

    def objective_function(self, particle_arr):
        new_seed = self.seed * (2 + int(len(self.particles) / self.n_particles))
        reset_seed(new_seed)
        particle = PartXGB()
        particle.estimators = int(particle_arr[0])
        particle.max_depth = int(particle_arr[1])
        particle.booster = self.BOOSTERS[int(particle_arr[2])]
        particle.reg_lambda = float(particle_arr[3])
        particle.reg_alpha = float(particle_arr[4])
        particle.seed = new_seed

        search = list(filter(lambda par:
                             par.estimators == particle.estimators and
                             par.max_depth == particle.max_depth and
                             par.booster == particle.booster and
                             par.reg_lambda == particle.reg_lambda and
                             par.reg_alpha == particle.reg_alpha, self.particles))

        if search:
            self.particles.append(search[0])
            return search[0].fitness

        x = self.dataset.drop("consumo", axis=1)
        y = self.dataset["consumo"]

        updater = "coord_descent" if particle.booster == "gblinear" else None
        model = XGBRegressor(device="cuda", random_state=self.seed,
                             n_estimators=particle.estimators,
                             max_depth=particle.max_depth,
                             booster=particle.booster,
                             reg_lambda=particle.reg_lambda,
                             reg_alpha=particle.reg_alpha,
                             updater=updater)

        cvs = []
        for i_train, i_test in TimeSeriesSplit(n_splits=5, test_size=1).split(x, y):
            x_train, x_test = x.iloc[i_train].to_cupy().get(), x.iloc[i_test].to_cupy().get()
            y_train, y_test = y.iloc[i_train].to_cupy().get(), y.iloc[i_test].to_cupy().get()

            model.fit(x_train, y_train)

            cvs.append(int(mean_absolute_error(y_test, model.predict(x_test))))
            del x_train, x_test, y_train, y_test

        particle.fitness = int(numpy.array(cvs).mean())

        self.particles.append(particle)

        del new_seed, x, y, cvs, i_train, i_test, model
        gc.collect()
        return particle.fitness

    def particles_dataframe(self):
        df = pd.DataFrame()
        for part in self.particles:
            df = pd.concat([df, pd.DataFrame({
                "N_estimators": part.estimators,
                "Max_depth": part.max_depth,
                "Booster": part.booster,
                "Lambda": part.reg_lambda,
                "Alpha": part.reg_alpha,
                "Seed": part.seed,
                "Fitness": part.fitness
            })])
        return df

    def iters_dataframe(self):
        df = pd.DataFrame()
        for part in self.iters:
            df = pd.concat([df, pd.DataFrame({
                "N_estimators": part.estimators,
                "Max_depth": part.max_depth,
                "Booster": part.booster,
                "Lambda": part.reg_lambda,
                "Alpha": part.reg_alpha,
                "Seed": part.seed,
                "Fitness": part.fitness
            })])
        return df

    def save_parts_csv(self):
        pd_df = self.particles_dataframe().to_pandas()
        pd_df.to_csv(f"results/PSO-XGB POP SEED {self.seed}.csv", sep=";", decimal=",", index=True)
        del pd_df

    def save_iters_csv(self):
        pd_df = self.iters_dataframe().to_pandas()
        pd_df.to_csv(f"results/PSO-XGB ITERS SEED {self.seed}.csv", sep=";", decimal=",", index=True)
        del pd_df


# %% md
# ### SVR
# %%
class PartSVR:
    def __init_(self):
        self.fitness = None
        self.seed = None
        self.c = 0
        self.epsilon = 0
        self.degree = 0
        self.kernel = None
        self.gamma = None


class PSOSVR:
    def __init__(self, dataset, n_particles, n_iters, seed=SEED):
        reset_seed(seed)
        self.seed = seed
        self.dataset = dataset
        self.n_particles = n_particles
        self.n_iters = n_iters
        self.particles = []
        self.iters = []
        self.KERNELS = ["poly", "rbf", "sigmoid"]
        self.run()

    def run(self):
        lower_bound = [0.001, 0.001, 0]
        uppper_bound = [300, 300, 2]
        bounds = (lower_bound, uppper_bound)

        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = GlobalBestPSO(n_particles=self.n_particles,
                                  dimensions=3,
                                  options=options,
                                  bounds=bounds)

        optimizer.optimize(self.get_fitness, iters=self.n_iters)
        self.particles = sorted(self.particles, key=lambda a: a.fitness)

    def get_fitness(self, parts):
        parts = np.round(parts)
        fit_lst = dask.compute([dask.delayed(self.objective_function)(parts[j]) for j in range(self.n_particles)])[0]
        self.particles = sorted(self.particles, key=lambda a: a.fitness)
        self.iters.append(self.particles[0])

        self.save_parts_csv()
        self.save_iters_csv()

        gc.collect()
        return fit_lst

    def objective_function(self, particle_arr):
        new_seed = self.seed * (2 + int(len(self.particles) / self.n_particles))
        reset_seed(new_seed)
        particle = PartSVR()
        particle.c = float(particle_arr[0])
        particle.epsilon = float(particle_arr[1])
        particle.kernel = self.KERNELS[int(particle_arr[2])]
        particle.seed = new_seed

        search = list(filter(lambda par:
                             par.c == particle.c and
                             par.epsilon == particle.epsilon and
                             par.degree == particle.degree and
                             par.kernel == particle.kernel and
                             par.gamma == particle.gamma, self.particles))

        if search:
            self.particles.append(search[0])
            return search[0].fitness

        x = self.dataset.drop("consumo", axis=1)
        y = self.dataset["consumo"]

        model = svr.SVR(C=particle.c,
                        epsilon=particle.epsilon,
                        kernel=particle.kernel)

        cvs = []
        for i_train, i_test in TimeSeriesSplit(n_splits=5, test_size=1).split(x, y):
            x_train, x_test = x.iloc[i_train].to_cupy().get(), x.iloc[i_test].to_cupy().get()
            y_train, y_test = y.iloc[i_train].to_cupy().get(), y.iloc[i_test].to_cupy().get()

            model.fit(x_train, y_train)

            cvs.append(int(mean_absolute_error(y_test, model.predict(x_test).get())))
            del x_train, x_test, y_train, y_test

        particle.fitness = int(numpy.array(cvs).mean())

        self.particles.append(particle)

        del new_seed, x, y, cvs, i_train, i_test, model
        gc.collect()
        return particle.fitness

    def particles_dataframe(self):
        df = pd.DataFrame()
        for part in self.particles:
            df = pd.concat([df, pd.DataFrame({
                "C": part.c,
                "Epsilon": part.epsilon,
                "Kernel": part.kernel,
                "Seed": part.seed,
                "Fitness": part.fitness
            })])
        return df

    def iters_dataframe(self):
        df = pd.DataFrame()
        for part in self.iters:
            df = pd.concat([df, pd.DataFrame({
                "C": part.c,
                "Epsilon": part.epsilon,
                "Kernel": part.kernel,
                "Seed": part.seed,
                "Fitness": part.fitness
            })])
        return df

    def save_parts_csv(self):
        pd_df = self.particles_dataframe().to_pandas()
        pd_df.to_csv(f"results/PSO-SVR POP SEED {self.seed}.csv", sep=";", decimal=",", index=True)
        del pd_df

    def save_iters_csv(self):
        pd_df = self.iters_dataframe().to_pandas()
        pd_df.to_csv(f"results/PSO-SVR ITERS SEED {self.seed}.csv", sep=";", decimal=",", index=True)
        del pd_df


# %% md
# ### LSTM
# %%
class PartLSTM:
    def __init_(self):
        self.fitness = None
        self.seed = None
        self.lstm_units = 0
        self.epochs = 0
        self.batch_size = 0
        self.lstm_activation = None
        self.bias = None


class PSOLSTM:
    def __init__(self, dataset, n_particles, n_iters, seed=SEED):
        reset_seed(seed)
        self.seed = seed
        self.dataset = dataset
        self.n_particles = n_particles
        self.n_iters = n_iters
        self.particles = []
        self.iters = []
        self.ACTIVATIONS = ["linear", "mish", "sigmoid", "softmax", "softplus", "softsign", "tanh", None]
        self.BIAS = [False, True]
        self.run()

    def run(self):
        lower_bound = [1, 1, 1, 0, 0]
        uppper_bound = [300, 100, 300, 7, 1]
        bounds = (lower_bound, uppper_bound)

        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = GlobalBestPSO(n_particles=self.n_particles,
                                  dimensions=5,
                                  options=options,
                                  bounds=bounds)

        optimizer.optimize(self.get_fitness, iters=self.n_iters)
        self.particles = sorted(self.particles, key=lambda a: a.fitness)

    def get_fitness(self, parts):
        parts = np.round(parts)
        fit_lst = dask.compute([dask.delayed(self.objective_function)(parts[j]) for j in range(self.n_particles)])[0]
        self.particles = sorted(self.particles, key=lambda a: a.fitness)
        self.iters.append(self.particles[0])

        self.save_parts_csv()
        self.save_iters_csv()

        tf.keras.backend.clear_session(True)
        return fit_lst

    def objective_function(self, particle_arr):
        new_seed = self.seed * (2 + int(len(self.particles) / self.n_particles))
        reset_seed(new_seed)
        particle = PartLSTM()
        particle.lstm_units = int(particle_arr[0])
        particle.epochs = int(particle_arr[1])
        particle.batch_size = int(particle_arr[2])
        particle.lstm_activation = self.ACTIVATIONS[int(particle_arr[3])]
        particle.bias = self.BIAS[int(particle_arr[4])]
        particle.seed = new_seed

        search = list(filter(lambda par:
                             par.lstm_units == particle.lstm_units and
                             par.epochs == particle.epochs and
                             par.batch_size == particle.batch_size and
                             par.lstm_activation == particle.lstm_activation and
                             par.bias == particle.bias, self.particles))

        if search:
            self.particles.append(search[0])
            return search[0].fitness

        x = self.dataset.drop("consumo", axis=1)
        y = self.dataset["consumo"]

        model = Sequential([
            Input((x.shape[1], 1)),
            LSTM(particle.lstm_units,
                 activation=particle.lstm_activation,
                 use_bias=particle.bias,
                 seed=self.seed),
            Dense(1),
        ])
        model.compile(loss='mse')

        cvs = []
        for i_train, i_test in TimeSeriesSplit(n_splits=5, test_size=1).split(x, y):
            x_train, x_test = x.iloc[i_train].to_cupy().get(), x.iloc[i_test].to_cupy().get()
            y_train, y_test = y.iloc[i_train].to_cupy().get(), y.iloc[i_test].to_cupy().get()

            model.fit(x_train, y_train, shuffle=False, verbose=False, epochs=particle.epochs,
                      batch_size=particle.batch_size)

            cvs.append(int(mean_absolute_error(y_test, model.predict(x_test)[0])))
            del x_train, x_test, y_train, y_test

        particle.fitness = int(numpy.array(cvs).mean())

        self.particles.append(particle)

        del new_seed, x, y, cvs, i_train, i_test, model
        gc.collect()
        return particle.fitness

    def particles_dataframe(self):
        df = pd.DataFrame()
        for part in self.particles:
            df = pd.concat([df, pd.DataFrame({
                "Units": part.lstm_units,
                "Epochs": part.epochs,
                "Batch Size": part.batch_size,
                "Activation": part.lstm_activation,
                "Bias": part.bias,
                "Seed": part.seed,
                "Fitness": part.fitness
            })])
        return df

    def iters_dataframe(self):
        df = pd.DataFrame()
        for part in self.iters:
            df = pd.concat([df, pd.DataFrame({
                "Units": part.lstm_units,
                "Epochs": part.epochs,
                "Batch Size": part.batch_size,
                "Activation": part.lstm_activation,
                "Bias": part.bias,
                "Seed": part.seed,
                "Fitness": part.fitness
            })])
        return df

    def save_parts_csv(self):
        pd_df = self.particles_dataframe().to_pandas()
        pd_df.to_csv(f"results/PSO-LSTM POP SEED {self.seed}.csv", sep=";", decimal=",", index=True)
        del pd_df

    def save_iters_csv(self):
        pd_df = self.iters_dataframe().to_pandas()
        pd_df.to_csv(f"results/PSO-LSTM ITERS SEED {self.seed}.csv", sep=";", decimal=",", index=True)
        del pd_df


# %% md
# # Aplicação dos Otimizadores
# ### Eletricidade
# %%

seeds = [1000, 2000, 3000]

if __name__ == "__main__":
    for seed in seeds:
        PSOXGB(df_electricity_selected, 30, 100, seed)
        PSOSVR(df_electricity_selected, 30, 100, seed)

        GARF(df_electricity_selected, 30, 100, seed)
        PSORF(df_electricity_selected, 30, 100, seed)

        GAXGB(df_electricity_selected, 30, 100, seed)

        GASVR(df_electricity_selected, 30, 100, seed)

        GALSTM(df_electricity_selected, 30, 100, seed)
        PSOLSTM(df_electricity_selected, 30, 100, seed)


    # %% md
# # Análise dos SHAP values dos modelos Otimizados
# %%

# %% md
# # Previsões
# ## Eletricidade
# ### 3 Passos à frente
# 
# %%
# reset_seed()
# x_electricity = df_electricity.drop("consumo", axis=1)
# y_electricity = df_electricity["consumo"]
# 
# xgb_electricity = XGBRegressor()
# rf_electricity = CudaRandomForest(n_streams=1, n_bins=x_electricity.shape[0])
# svr_electricity = svr.SVR()
# lstm_electricity = Sequential([
#     Input((x_electricity.shape[1], 1)),
#     LSTM(30, activation='relu', seed=SEED),
#     Dense(1),
# ])
# lstm_electricity.compile(loss='mse')
# 
# x_train, x_test, y_train, y_test = train_test_split(x_electricity, y_electricity, test_size=3, shuffle=False)
# 
# cvs_electricity = pd.DataFrame()
# for i_train, i_test in TimeSeriesSplit(n_splits=5, test_size=1).split(x_train, y_train):
#     kx_train, kx_test = x_train.iloc[i_train].to_numpy(), x_train.iloc[i_test].to_numpy()
#     ky_train, ky_test = y_train.iloc[i_train].to_numpy(), y_train.iloc[i_test].to_numpy()
# 
#     xgb_electricity.fit(kx_train, ky_train)
#     rf_electricity.fit(kx_train, ky_train)
#     svr_electricity.fit(kx_train, ky_train)
#     lstm_electricity.fit(kx_train, ky_train, shuffle=False, verbose=False, epochs=1)
#     cvs_electricity = pd.concat([cvs_electricity, pd.DataFrame({
#         "XGB": mean_absolute_percentage_error(xgb_electricity.predict(kx_test), ky_test),
#         "RF": mean_absolute_percentage_error(rf_electricity.predict(kx_test), ky_test),
#         "SVR": mean_absolute_percentage_error(svr_electricity.predict(kx_test), ky_test),
#         "LSTM": mean_absolute_percentage_error(lstm_electricity.predict(kx_test), ky_test)
#     })])
# 
# pred_xgb_electricity = []
# for i_test in range(len(x_test)):
#     sx_test = x_test.iloc[[i_test]]
# 
#     for climatic_column in df_climatic.drop(["ano", "mes"], axis=1).columns:
#         sx_test.at[sx_test.index, climatic_column] = \
#             x_electricity.at[(sx_test.index - pd.DateOffset(years=1)), climatic_column].to_numpy()[0][0]
#     for lag in range(i_test + 1):
#         if 'consumo_LAG_' + "{:02d}".format(lag) in sx_test.columns:
#             sx_test['consumo_LAG_' + "{:02d}".format(lag)] = pred_xgb_electricity[-lag]
# 
#     pred_xgb_electricity.append(xgb_electricity.predict(sx_test.to_numpy())[0])
# 
# pred_rf_electricity = []
# for i_test in range(len(x_test)):
#     sx_test = x_test.iloc[[i_test]]
# 
#     for climatic_column in df_climatic.drop(["ano", "mes"], axis=1).columns:
#         sx_test.at[sx_test.index, climatic_column] = \
#             x_electricity.at[(sx_test.index - pd.DateOffset(years=1)), climatic_column].to_numpy()[0][0]
#     for lag in range(i_test + 1):
#         if 'consumo_LAG_' + "{:02d}".format(lag) in sx_test.columns:
#             sx_test['consumo_LAG_' + "{:02d}".format(lag)] = pred_rf_electricity[-lag]
# 
#     pred_rf_electricity.append(rf_electricity.predict(sx_test.to_numpy())[0])
# 
# pred_svr_electricity = []
# for i_test in range(len(x_test)):
#     sx_test = x_test.iloc[[i_test]]
# 
#     for climatic_column in df_climatic.drop(["ano", "mes"], axis=1).columns:
#         sx_test.at[sx_test.index, climatic_column] = \
#             x_electricity.at[(sx_test.index - pd.DateOffset(years=1)), climatic_column].to_numpy()[0][0]
#     for lag in range(i_test + 1):
#         if 'consumo_LAG_' + "{:02d}".format(lag) in sx_test.columns:
#             sx_test['consumo_LAG_' + "{:02d}".format(lag)] = pred_svr_electricity[-lag]
# 
#     pred_svr_electricity.append(svr_electricity.predict(sx_test.to_numpy())[0])
# 
# pred_lstm_electricity = []
# for i_test in range(len(x_test)):
#     sx_test = x_test.iloc[[i_test]]
# 
#     for climatic_column in df_climatic.drop(["ano", "mes"], axis=1).columns:
#         sx_test.at[sx_test.index, climatic_column] = \
#             x_electricity.at[(sx_test.index - pd.DateOffset(years=1)), climatic_column].to_numpy()[0][0]
#     for lag in range(i_test + 1):
#         if 'consumo_LAG_' + "{:02d}".format(lag) in sx_test.columns:
#             sx_test['consumo_LAG_' + "{:02d}".format(lag)] = pred_lstm_electricity[-lag]
# 
#     pred_lstm_electricity.append(lstm_electricity.predict(sx_test.to_numpy())[0])

# %% md
# ### 6 Passos à frente
# %%
# reset_seed()
# x_electricity = df_electricity.drop("consumo", axis=1)
# y_electricity = df_electricity["consumo"]
# 
# xgb_electricity = XGBRegressor()
# rf_electricity = CudaRandomForest(n_streams=1, n_bins=x_electricity.shape[0])
# svr_electricity = svr.SVR()
# lstm_electricity = Sequential([
#     Input((x_electricity.shape[1], 1)),
#     LSTM(30, activation='relu', seed=SEED),
#     Dense(1),
# ])
# lstm_electricity.compile(loss='mse')
# 
# x_train, x_test, y_train, y_test = train_test_split(x_electricity, y_electricity, test_size=6, shuffle=False)
# 
# cvs_electricity = pd.DataFrame()
# for i_train, i_test in TimeSeriesSplit(n_splits=5, test_size=1).split(x_train, y_train):
#     kx_train, kx_test = x_train.iloc[i_train].to_numpy(), x_train.iloc[i_test].to_numpy()
#     ky_train, ky_test = y_train.iloc[i_train].to_numpy(), y_train.iloc[i_test].to_numpy()
# 
#     xgb_electricity.fit(kx_train, ky_train)
#     rf_electricity.fit(kx_train, ky_train)
#     svr_electricity.fit(kx_train, ky_train)
#     lstm_electricity.fit(kx_train, ky_train, shuffle=False, verbose=False, epochs=1)
#     cvs_electricity = pd.concat([cvs_electricity, pd.DataFrame({
#         "XGB": mean_absolute_percentage_error(xgb_electricity.predict(kx_test), ky_test),
#         "RF": mean_absolute_percentage_error(rf_electricity.predict(kx_test), ky_test),
#         "SVR": mean_absolute_percentage_error(svr_electricity.predict(kx_test), ky_test),
#         "LSTM": mean_absolute_percentage_error(lstm_electricity.predict(kx_test), ky_test)
#     })])
# 
# pred_xgb_electricity = []
# for i_test in range(len(x_test)):
#     sx_test = x_test.iloc[[i_test]]
# 
#     for climatic_column in df_climatic.drop(["ano", "mes"], axis=1).columns:
#         sx_test.at[sx_test.index, climatic_column] = \
#             x_electricity.at[(sx_test.index - pd.DateOffset(years=1)), climatic_column].to_numpy()[0][0]
#     for lag in range(i_test + 1):
#         if 'consumo_LAG_' + "{:02d}".format(lag) in sx_test.columns:  
#             sx_test['consumo_LAG_' + "{:02d}".format(lag)] = pred_xgb_electricity[-lag]
# 
#     pred_xgb_electricity.append(xgb_electricity.predict(sx_test.to_numpy())[0])
# 
# pred_rf_electricity = []
# for i_test in range(len(x_test)):
#     sx_test = x_test.iloc[[i_test]]
# 
#     for climatic_column in df_climatic.drop(["ano", "mes"], axis=1).columns:
#         sx_test.at[sx_test.index, climatic_column] = \
#             x_electricity.at[(sx_test.index - pd.DateOffset(years=1)), climatic_column].to_numpy()[0][0]
#     for lag in range(i_test + 1):
#         if 'consumo_LAG_' + "{:02d}".format(lag) in sx_test.columns:  
#             sx_test['consumo_LAG_' + "{:02d}".format(lag)] = pred_rf_electricity[-lag]
# 
#     pred_rf_electricity.append(rf_electricity.predict(sx_test.to_numpy())[0])
# 
# pred_svr_electricity = []
# for i_test in range(len(x_test)):
#     sx_test = x_test.iloc[[i_test]]
# 
#     for climatic_column in df_climatic.drop(["ano", "mes"], axis=1).columns:
#         sx_test.at[sx_test.index, climatic_column] = \
#             x_electricity.at[(sx_test.index - pd.DateOffset(years=1)), climatic_column].to_numpy()[0][0]
#     for lag in range(i_test + 1):
#         if 'consumo_LAG_' + "{:02d}".format(lag) in sx_test.columns:  
#             sx_test['consumo_LAG_' + "{:02d}".format(lag)] = pred_svr_electricity[-lag]
# 
#     pred_svr_electricity.append(svr_electricity.predict(sx_test.to_numpy())[0])
# 
# pred_lstm_electricity = []
# for i_test in range(len(x_test)):
#     sx_test = x_test.iloc[[i_test]]
# 
#     for climatic_column in df_climatic.drop(["ano", "mes"], axis=1).columns:
#         sx_test.at[sx_test.index, climatic_column] = \
#             x_electricity.at[(sx_test.index - pd.DateOffset(years=1)), climatic_column].to_numpy()[0][0]
#     for lag in range(i_test + 1):
#         if 'consumo_LAG_' + "{:02d}".format(lag) in sx_test.columns:  
#             sx_test['consumo_LAG_' + "{:02d}".format(lag)] = pred_lstm_electricity[-lag]
# 
#     pred_lstm_electricity.append(lstm_electricity.predict(sx_test.to_numpy())[0])

# %% md
# ### 12 Passos à frente
# %%
# reset_seed()
# x_electricity = df_electricity.drop("consumo", axis=1)
# y_electricity = df_electricity["consumo"]
# 
# xgb_electricity = XGBRegressor()
# rf_electricity = CudaRandomForest(n_streams=1, n_bins=x_electricity.shape[0])
# svr_electricity = svr.SVR()
# lstm_electricity = Sequential([
#     Input((x_electricity.shape[1], 1)),
#     LSTM(30, activation='relu', seed=SEED),
#     Dense(1),
# ])
# lstm_electricity.compile(loss='mse')
# 
# x_train, x_test, y_train, y_test = train_test_split(x_electricity, y_electricity, test_size=12, shuffle=False)
# 
# cvs_electricity = pd.DataFrame()
# for i_train, i_test in TimeSeriesSplit(n_splits=5, test_size=1).split(x_train, y_train):
#     kx_train, kx_test = x_train.iloc[i_train].to_numpy(), x_train.iloc[i_test].to_numpy()
#     ky_train, ky_test = y_train.iloc[i_train].to_numpy(), y_train.iloc[i_test].to_numpy()
# 
#     xgb_electricity.fit(kx_train, ky_train)
#     rf_electricity.fit(kx_train, ky_train)
#     svr_electricity.fit(kx_train, ky_train)
#     lstm_electricity.fit(kx_train, ky_train, shuffle=False, verbose=False, epochs=1)
#     cvs_electricity = pd.concat([cvs_electricity, pd.DataFrame({
#         "XGB": mean_absolute_percentage_error(xgb_electricity.predict(kx_test), ky_test),
#         "RF": mean_absolute_percentage_error(rf_electricity.predict(kx_test), ky_test),
#         "SVR": mean_absolute_percentage_error(svr_electricity.predict(kx_test), ky_test),
#         "LSTM": mean_absolute_percentage_error(lstm_electricity.predict(kx_test), ky_test)
#     })])
# 
# pred_xgb_electricity = []
# for i_test in range(len(x_test)):
#     sx_test = x_test.iloc[[i_test]]
# 
#     for climatic_column in df_climatic.drop(["ano", "mes"], axis=1).columns:
#         sx_test.at[sx_test.index, climatic_column] = \
#             x_electricity.at[(sx_test.index - pd.DateOffset(years=1)), climatic_column].to_numpy()[0][0]
#     for lag in range(i_test + 1):
#         if 'consumo_LAG_' + "{:02d}".format(lag) in sx_test.columns:  
#             sx_test['consumo_LAG_' + "{:02d}".format(lag)] = pred_xgb_electricity[-lag]
# 
#     pred_xgb_electricity.append(xgb_electricity.predict(sx_test.to_numpy())[0])
# 
# pred_rf_electricity = []
# for i_test in range(len(x_test)):
#     sx_test = x_test.iloc[[i_test]]
# 
#     for climatic_column in df_climatic.drop(["ano", "mes"], axis=1).columns:
#         sx_test.at[sx_test.index, climatic_column] = \
#             x_electricity.at[(sx_test.index - pd.DateOffset(years=1)), climatic_column].to_numpy()[0][0]
#     for lag in range(i_test + 1):
#         if 'consumo_LAG_' + "{:02d}".format(lag) in sx_test.columns:  
#             sx_test['consumo_LAG_' + "{:02d}".format(lag)] = pred_rf_electricity[-lag]
# 
#     pred_rf_electricity.append(rf_electricity.predict(sx_test.to_numpy())[0])
# 
# pred_svr_electricity = []
# for i_test in range(len(x_test)):
#     sx_test = x_test.iloc[[i_test]]
# 
#     for climatic_column in df_climatic.drop(["ano", "mes"], axis=1).columns:
#         sx_test.at[sx_test.index, climatic_column] = \
#             x_electricity.at[(sx_test.index - pd.DateOffset(years=1)), climatic_column].to_numpy()[0][0]
#     for lag in range(i_test + 1):
#         if 'consumo_LAG_' + "{:02d}".format(lag) in sx_test.columns:  
#             sx_test['consumo_LAG_' + "{:02d}".format(lag)] = pred_svr_electricity[-lag]
# 
#     pred_svr_electricity.append(svr_electricity.predict(sx_test.to_numpy())[0])
# 
# pred_lstm_electricity = []
# for i_test in range(len(x_test)):
#     sx_test = x_test.iloc[[i_test]]
# 
#     for climatic_column in df_climatic.drop(["ano", "mes"], axis=1).columns:
#         sx_test.at[sx_test.index, climatic_column] = \
#             x_electricity.at[(sx_test.index - pd.DateOffset(years=1)), climatic_column].to_numpy()[0][0]
#     for lag in range(i_test + 1):
#         if 'consumo_LAG_' + "{:02d}".format(lag) in sx_test.columns:  
#             sx_test['consumo_LAG_' + "{:02d}".format(lag)] = pred_lstm_electricity[-lag]
# 
#     pred_lstm_electricity.append(lstm_electricity.predict(sx_test.to_numpy())[0])
