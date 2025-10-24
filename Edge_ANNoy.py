# -*- coding: utf-8 -*-
from sympy import continued_fraction_periodic
from Node import Node
import torch
import numpy as np
from tqdm import tqdm

if torch.cuda.is_available():
	device = torch.device("cuda")  # 使用 GPU
else:
	device = torch.device("cpu")  # 使用 CPU


class Edge_ANNoy:
	def __init__(self,
				 num_of_level_value,
				 All_tensor,
				 data_ids,
				 dis_space="euclidean"):
		self.num_of_level = num_of_level_value
		# 将数据 tensor 存储在 CPU 内存中，仅在需要时移动到 GPU,可解决显存不足问题
		self.tensor_cpu = All_tensor  # 数据存储在 CPU
		self.root = Node(data_ids)
		self.dis_space = dis_space
		# self.progress_bar = tqdm(total=len(data_ids), desc='Building Edge-ANNoy', unit='vec')  # 新增进度条
		self.test = 0
		print(device)

	def _get_tensor_on_device(self, ids):
		return self.tensor_cpu[ids].to(device)

	def find_nearest_point(self, id_list, vector, seed):
		selected_vectors_gpu = self._get_tensor_on_device(id_list) # 将需要的数据子集移动到 GPU
		vector_gpu = vector.to(device) # 确保 vector 也在 GPU 上
		para = -1
		if self.dis_space == "euclidean":
			para = 2
		elif self.dis_space == "manhattan":
			para = 1
		else:
			para = 2
		distances = torch.norm(selected_vectors_gpu - vector_gpu, dim=1, p=para)
		_, min2_index = torch.topk(distances, k=2, largest=False)
		if id_list[min2_index[0]] == seed:
			nearest_id = id_list[min2_index[1]]
		else:
			nearest_id = id_list[min2_index[0]]
		return nearest_id # 返回的是 cpu id list 的 index， 不需要放回 cpu

	def calculate_median_distance(self, data_ids, vector_left, vector_right):
		selected_vectors_gpu = self._get_tensor_on_device(data_ids) # 将需要的数据子集移动到 GPU
		vector_left_gpu = vector_left.to(device) # 确保 vector 也在 GPU 上
		vector_right_gpu = vector_right.to(device) # 确保 vector 也在 GPU 上

		para = -1
		if self.dis_space == "euclidean":
			para = 2
		elif self.dis_space == "manhattan":
			para = 1
		else:
			para = 2

		distances_left = torch.norm(selected_vectors_gpu - vector_left_gpu, dim=1, p=para)
		median_left = torch.median(distances_left)
		distances_right = torch.norm(selected_vectors_gpu - vector_right_gpu, dim=1, p=para)
		median_right = torch.median(distances_right)
		median = (median_left + median_right) / 2
		return median.cpu() # 中位数距离放回 CPU

	def move_until_balance(self, node):
		para = 2 if self.dis_space == "euclidean" else 1
		
		best_seeds = None
		# max_hyperplane_dist = 0.0
		
		for _ in range(1):       # 可尝试多组随机种子以寻找最佳分割
			seeds = np.random.choice(node.data_ids, size=2, replace=False)
			# hyperplane_dist = self.compute_hyperplane_distances(seeds, node.data_ids)
			# if hyperplane_dist > max_hyperplane_dist:
				# max_hyperplane_dist = hyperplane_dist
			best_seeds = seeds
		
		# 使用最优的种子点继续后续操作
		seeds = best_seeds
		vectors_gpu = self._get_tensor_on_device(node.data_ids)
		seed_vectors_gpu = self._get_tensor_on_device(seeds)
		
		distances_left = torch.norm(vectors_gpu - seed_vectors_gpu[0], p=para, dim=1).cpu()
		distances_right = torch.norm(vectors_gpu - seed_vectors_gpu[1], p=para, dim=1).cpu()
		left_ids = node.data_ids[distances_left < distances_right]
		right_ids = node.data_ids[distances_left >= distances_right]
		
		seeds_num = [len(left_ids), len(right_ids)]
		x = min(seeds_num[0], seeds_num[1]) / max(seeds_num[0], seeds_num[1])
		best_set = [seeds, left_ids, right_ids, x]
		index_left = seeds[0]
		index_right = seeds[1]
		penalty = 1.0
		enhance_ratio = 1.0
		balance_history = [x]  # 初始化当前平衡比
		dynamic_factor = 1.0	# 动态调整因子
		i = 0
		while True:
			if i == 5:
				break
			# print(i)
			i += 1
			if len(balance_history) >= 2:
				changes = [balance_history[i+1]-balance_history[i] for i in range(len(balance_history)-1)]
				change_rate = np.mean(changes)
				# 动态调整逻辑(局部变量，不影响其他节点)
				if abs(change_rate) < 0.02:
					dynamic_factor *= 3
				if abs(change_rate) < 0.1: 
					dynamic_factor *= 1.5
				elif abs(change_rate) > 0.2:
					dynamic_factor *= 0.7
			org_seeds_num = [seeds_num[0], seeds_num[1]]
			vector_left_gpu = self._get_tensor_on_device([index_left])
			vector_right_gpu = self._get_tensor_on_device([index_right])
			median_distance = self.calculate_median_distance(node.data_ids, vector_left_gpu, vector_right_gpu)
			direction_vector_gpu = vector_right_gpu - vector_left_gpu if len(left_ids) < len(right_ids) else vector_left_gpu - vector_right_gpu
			direction_vector_gpu = direction_vector_gpu / torch.norm(direction_vector_gpu) * median_distance / 2
			x = min(len(right_ids), len(left_ids)) / max(len(right_ids), len(left_ids))
			# weight = 0.2 + 1.1 / (1.3 + np.exp(15 * (x - 0.55)))  # 可调整S曲线参数
			weight = 0.5 * np.exp(2 - x)
			vectors_gpu = self._get_tensor_on_device(node.data_ids)
			j = 0
			while True:
				if best_set[3] > 0.65 or j == 5:
					break
				j += 1

				vector_left_gpu = vector_left_gpu.to(device)
				vector_right_gpu = vector_right_gpu.to(device)
				index_left = self.find_nearest_point(node.data_ids, vector_left_gpu + weight * penalty * enhance_ratio * direction_vector_gpu, seeds[0]) if seeds_num[0] > seeds_num[1] else seeds[0]
				index_right = self.find_nearest_point(node.data_ids, vector_right_gpu + weight * penalty * enhance_ratio * direction_vector_gpu, seeds[1]) if seeds_num[0] < seeds_num[1] else seeds[1]
				if index_left == seeds[0] or index_right == seeds[1]:
					enhance_ratio *= 2
					continue

				if index_left != index_right:
					break
				enhance_ratio *= 0.5
				if enhance_ratio < 1e-6:
					break
			if index_left == index_right:
				break
			vectors_gpu = self._get_tensor_on_device(node.data_ids)
			seed_vectors_gpu = self._get_tensor_on_device([index_left.item(), index_right.item()])
			distances_left = torch.norm(vectors_gpu - seed_vectors_gpu[0], p=para, dim=1).cpu()
			distances_right = torch.norm(vectors_gpu - seed_vectors_gpu[1], p=para, dim=1).cpu()

			left_ids = node.data_ids[distances_left < distances_right]
			right_ids = node.data_ids[distances_left >= distances_right]
			seeds = torch.tensor([index_left, index_right])
			seeds_num = [len(left_ids), len(right_ids)]
			x = min(seeds_num[0], seeds_num[1]) / max(seeds_num[0], seeds_num[1])
			# print(seeds_num[0], " ",seeds_num[1])
			# print(x)
			if x > best_set[3]:
				best_set = [seeds, left_ids, right_ids, x]
			if best_set[3] > 0.65:
				break
			# current_seed_pair = (seeds[0].item(), seeds[1].item())
			# penalty = self.calculate_penalty(org_seeds_num, seeds_num)
			penalty *= self.calculate_penalty(org_seeds_num, seeds_num) * dynamic_factor
			balance_history.append(x)

		seed_vector1_gpu = self._get_tensor_on_device([best_set[0][0]])
		seed_vector2_gpu = self._get_tensor_on_device([best_set[0][1]])
		
		# 计算超平面法向量和中心点
		normal_vector = seed_vector2_gpu - seed_vector1_gpu
		mid_point = (seed_vector1_gpu + seed_vector2_gpu) / 2.0

		# 计算所有点到超平面的距离
		all_vectors_gpu = self._get_tensor_on_device(node.data_ids)
		normal_vector_unsq = normal_vector.unsqueeze(-1)
		constant_term = torch.dot(normal_vector.flatten(), mid_point.flatten())
		norm = torch.linalg.norm(normal_vector)
		
		# 计算所有点的有符号距离
		signed_distances = (torch.matmul(all_vectors_gpu, normal_vector_unsq).squeeze(-1) - constant_term) / norm
		signed_distances = signed_distances.squeeze()  
		
		# 直接使用中位数作为分割点
		median_distance = torch.median(signed_distances)
		
		# 使用中位数重新分配点
		mask = signed_distances < median_distance
		left_ids = node.data_ids[mask.cpu()]
		right_ids = node.data_ids[~mask.cpu()]

		# 更新best_set
		best_set = [best_set[0], left_ids, right_ids, min(len(left_ids), len(right_ids)) / max(len(left_ids), len(right_ids))]

		return best_set[1], best_set[2], best_set[0], median_distance.item()
	
	def compute_hyperplane_distances(self, seeds, all_ids):
		seed_vector1_gpu = self._get_tensor_on_device([seeds[0]])
		seed_vector2_gpu = self._get_tensor_on_device([seeds[1]])
		
		# 计算超平面参数
		mid_point = (seed_vector1_gpu + seed_vector2_gpu) / 2.0
		normal_vector = seed_vector2_gpu - seed_vector1_gpu
		norm = torch.linalg.norm(normal_vector)
		constant_term = torch.dot(normal_vector.flatten(), mid_point.flatten())
		
		all_vectors_gpu = self._get_tensor_on_device(all_ids)
		normal_vector_unsq = normal_vector.unsqueeze(-1)
		signed_distances = (torch.matmul(all_vectors_gpu, normal_vector_unsq).squeeze(-1) - constant_term) / norm
		
		# 计算中位数
		median_distance = torch.median(signed_distances)
		
		# 计算所有距离与中位数的绝对差之和
		abs_diff_sum = torch.sum(torch.abs(signed_distances - median_distance)).item()
		
		return abs_diff_sum

	def calculate_penalty(self, org_seeds, cur_seeds):
		penalty = 1.0
		org_co = org_seeds[0] / org_seeds[1]
		new_co = cur_seeds[0] / cur_seeds[1]
		if org_co <= 1.0 and new_co <= 1.0:
			if org_co <= new_co:
				penalty = 1.0
			else:
				penalty = 2.5
		elif org_co > 1.0 and new_co > 1.0:
			if org_co >= new_co:
				penalty = 1.0
			else:
				penalty = 2.5
		else:
			dis_org_to_aim = abs(org_co - 1.0)
			dis_new_to_aim = abs(new_co - 1.0)
			if dis_org_to_aim < dis_new_to_aim:
				penalty = 1.4
			else:
				penalty = 0.6
		return penalty

	def build(self):
		self.insert(self.root)
		# self.progress_bar.close()  # 构建完成后关闭进度条

	def insert(self, node):
		if (len(node.data_ids) <= self.num_of_level):  # 叶子节点条件判断
			# self.progress_bar.update(len(node.data_ids))  # 更新进度条
			return None

		left_ids, right_ids, seeds, b = self.move_until_balance(node)
		ratio = max(len(left_ids), len(right_ids)) / min(len(left_ids), len(right_ids))
		node.left_child = Node(left_ids)
		node.right_child = Node(right_ids)
		node.b = b
		node.nil = [seeds[0], seeds[1]]
		node.id_list = []
		self.insert(node.left_child)
		self.insert(node.right_child)

	def store_model(self, node, node_file, id_file):
		def _store_and_calc_balance(n):
			if n.left_child is None and n.right_child is None:
				# 叶子节点：写空行，写id
				with open(node_file, 'a') as nf, open(id_file, 'a') as idf:
					nf.write('\n')
					idf.write(' '.join(map(str, n.data_ids.tolist())) + '\n')
				return len(n.data_ids)  # 叶子节点无左右子树大小
			# 非叶子节点：写哨兵
			with open(node_file, 'a') as nf:
				nf.write(f"{n.nil[0].item()} {n.nil[1].item()} {n.b}\n")
			left_count = _store_and_calc_balance(n.left_child)
			right_count = _store_and_calc_balance(n.right_child)
			
			total_count = left_count + right_count
			return total_count
		total = _store_and_calc_balance(node)
		print(total)