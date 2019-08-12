import numpy as np
import random
import math
import time
import gc
import os
import utils
from env import Env

class RandomAgent():
    def __init__(self, config):
        self.config = config
        self.episode_length = int(self.config['META']['EPISODE_LENGTH'])
        self.sample_episodes_per_batch = int(self.config['TPGR']['SAMPLE_EPISODES_PER_BATCH'])
        self.sample_users_per_batch = int(self.config['TPGR']['SAMPLE_USERS_PER_BATCH'])
        self.boundary_rating = float(self.config['ENV']['BOUNDARY_RATING'])
        self.eval_batch_size = int(self.config['TPGR']['EVAL_BATCH_SIZE'])
        self.train_batch_size = self.sample_episodes_per_batch * self.sample_users_per_batch

        self.result_file_path = '../data/result/result_log/' + time.strftime('%Y%m%d%H%M%S') + '_' + self.config['ENV']['ALPHA'] + '_' + self.config['ENV']['BETA'] + '_' + self.config['ENV']['RATING_FILE']

        self.forward_env = Env(self.config)
        self.user_num, self.item_num, self.r_matrix, self.user_to_rele_num, genre_package = self.forward_env.get_init_data()
        self.boundry_user_id = self.forward_env.boundry_user_id

        self.env = [Env(self.config, self.user_num, self.item_num, self.r_matrix, self.user_to_rele_num, genre_package) for i in range(max(self.train_batch_size, self.eval_batch_size * int(math.ceil(self.user_num / self.eval_batch_size))))]

        self.storage = []
        self.training_steps = 0


    def update_avalable_items(self, sampled_items):
        self.history = []
        for i in range(len(sampled_items)):
            self.history.append(set([sampled_items[i]]))

    def sample(self):
        result = []
        for i in range(self.eval_batch_size):
            while True:
                item_id = random.randint(0, self.item_num - 1)
                if item_id not in self.history[i]:
                    break
            self.history[i].add(item_id)
            result.append(item_id)
        return result

    def _get_initial_ars(self, batch_size):
        result = [[[]], [[]], [[]]]
        for i in range(batch_size):
            item_id = random.randint(0, self.item_num - 1)
            reward = self.env[i].get_reward(item_id)
            result[0][0].append(item_id)
            result[1][0].append(reward[0])
            result[2][0].append((self.env[i].get_statistic()))
        return result

    def evaluate(self):
        eval_step_num = int(math.ceil(self.user_num / self.eval_batch_size))
        for i in range(0, self.eval_batch_size * eval_step_num):
            self.env[i].reset(i % self.user_num)
        ars = self._get_initial_ars(self.eval_batch_size * eval_step_num)

        for s in range(eval_step_num):
            start = s * self.eval_batch_size
            end = (s + 1) * self.eval_batch_size
            self.update_avalable_items(ars[0][0][start:end])
            step_count = 0
            stop_flag = False
            while True:
                sampled_action = self.sample()
                
                step_count += 1
                if len(ars[0]) == step_count:
                    ars[0].append([])
                    ars[1].append([])
                    ars[2].append([])
                for j in range(self.eval_batch_size):
                    reward = self.env[start + j].get_reward(sampled_action[j])
                    ars[0][step_count].append(sampled_action[j])
                    ars[1][step_count].append(reward[0])
                    ars[2][step_count].append(self.env[start + j].get_statistic())
                    if reward[1]:
                        stop_flag = True
                if stop_flag:
                    break
        
        reward_list = np.transpose(np.array(ars[1]))[:self.user_num]
        train_ave_reward = np.mean(reward_list[:self.boundry_user_id])
        test_ave_reward = np.mean(reward_list[self.boundry_user_id:self.user_num])
        
        tp_list = []
        rele_list = []
        for j in range(self.user_num):
            self.forward_env.reset(j)
            ratings = [self.forward_env.get_rating(ars[0][k][j]) for k in range(0, len(ars[0]))]
            tp = len(list(filter(lambda x: x >=self.boundary_rating, ratings)))
            tp_list.append(tp)
            rele_item_num = self.forward_env.get_relevant_item_num()
            rele_list.append(rele_item_num)

        precision = np.array(tp_list) / self.episode_length
        recall = np.array(tp_list) / (np.array(rele_list) + 1e-20)
        f1 = (2 * precision * recall) / (precision + recall + 1e-20)

        train_ave_precision = np.mean(precision[:self.boundry_user_id])
        train_ave_recall = np.mean(recall[:self.boundry_user_id])
        train_ave_f1 = np.mean(f1[:self.boundry_user_id])
        test_ave_precision = np.mean(precision[self.boundry_user_id:self.user_num])
        test_ave_recall = np.mean(recall[self.boundry_user_id:self.user_num])
        test_ave_f1 = np.mean(f1[self.boundry_user_id:self.user_num])
        ave_rmse = 0

        # save the result
        self.storage.append([train_ave_reward, train_ave_precision, train_ave_recall, train_ave_f1, test_ave_reward, test_ave_precision, test_ave_recall, test_ave_f1, ave_rmse])
        utils.pickle_save(self.storage, self.result_file_path)

        print('training step: %d' % (self.training_steps))
        print('\ttrain average reward over step: %2.4f, precision@%d: %.4f, recall@%d: %.4f, f1@%d: %.4f' % (train_ave_reward, self.episode_length, train_ave_precision, self.episode_length, train_ave_recall, self.episode_length, train_ave_f1))
        print('\ttest  average reward over step: %2.4f, precision@%d: %.4f, recall@%d: %.4f, f1@%d: %.4f' % (test_ave_reward, self.episode_length, test_ave_precision, self.episode_length, test_ave_recall, self.episode_length, test_ave_f1))
        print('\taverage rmse over train and test: %3.6f' % (ave_rmse))

        del ars
        gc.collect()        