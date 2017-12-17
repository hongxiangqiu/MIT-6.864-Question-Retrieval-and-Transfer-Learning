from datetime import datetime
import time
from src import cnn
from src.common import *

torch.cuda.set_device(0)

embeddings_p200 = EmbeddingsP200("./data_local/vectors_pruned.200.txt")
corpus = Corpus("./data_local/text_tokenized.txt", max_text_length=100)
train_data = TrainingData("./data_local/train_random.txt")
dev_data = read_android_annotations('./data_local/dev.txt')
dev_title_emb_list, dev_text_emb_list = batch_to_emb(dev_data, corpus=corpus, n_neg_samples=20, embeddings=embeddings_p200, train=False)

save_to_file = False
n_neg_samples = 20
hidden_dim = 667
batch_size = 10
batch_i = 0
tot_N = len(train_data.train_items)
n_batches = int(tot_N/batch_size)+1
n_cand = n_neg_samples+2
n_cand_dev = 21
emb_size = 200

kernel_size = 3
model = cnn.CNNQAModel(hidden_dim=hidden_dim, kernel_size=kernel_size, cand_size=n_cand, emb_size=emb_size, cand_size_dev=n_cand_dev, cuda=True)
max_margin_loss = torch.nn.MultiMarginLoss(margin = 0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

now = datetime.now()
session_id = "{}{:02d}{:02d}_{:02d}{:02d}_".format(now.year, now.month, now.day, now.hour, now.minute)+str(int(time.time()))
result_folder = "results_cnn/"+ session_id
output_file = os.path.join(result_folder, "output.txt")
printf = get_printf(save_to_file=save_to_file, output_file=output_file)

for n_epoch in range(6):
	train_data.shuffle()
	printf("Cur Epoch:", n_epoch)
	for batch_i in range(n_batches):
		optimizer.zero_grad()
				
		cur_batch = train_data.train_items[batch_size*batch_i : min(batch_size*batch_i+batch_size, tot_N)]
		
		# print(batch_i, n_batches)

		cur_batch_size = len(cur_batch)
		title_emb_list, text_emb_list = batch_to_emb(cur_batch, corpus=corpus, n_neg_samples=n_neg_samples, embeddings=embeddings_p200)
		text_emb, text_lengths, text_rev = pad_and_sort(text_emb_list)
		title_emb, title_lengths, title_rev = pad_and_sort(title_emb_list)

		text_rev = cuda(text_rev)
		title_rev = cuda(title_rev)
		
		title_emb_var = to_variable(title_emb)
		text_emb_var = to_variable(text_emb)
		title_length_var = to_variable(torch.from_numpy(title_lengths).float())
		text_length_var = to_variable(torch.from_numpy(text_lengths).float())

		cos_sim = model(title_emb_var, title_length_var, title_rev, text_emb_var, text_length_var, text_rev, cur_batch_size)
		const_y = to_variable(torch.from_numpy(np.array([0]*len(cur_batch))))
		# const_y = torch.autograd.Variable(torch.from_numpy(np.array([1]*len(cur_batch))), requires_grad=False)
		loss = max_margin_loss(cos_sim, const_y)
		loss.backward()
		optimizer.step()

		if (batch_i+1) % 50 == 0:
			printf ('Epoch %d  batch %d, Loss: %.4f' % (n_epoch+1, batch_i+1, loss.data[0]))

		if batch_i == n_batches-1:
			# Dev data, feed into model 1 by 1
			dev_pred = [ ]
			for dev_i in range(len(dev_title_emb_list)):
				title_emb = dev_title_emb_list[dev_i]
				text_emb = dev_text_emb_list[dev_i]

				text_emb, text_lengths, text_rev = pad_and_sort([text_emb])
				title_emb, title_lengths, title_rev = pad_and_sort([title_emb])

				text_rev = cuda(text_rev)
				title_rev = cuda(title_rev)

				title_emb_var = to_variable(title_emb)
				text_emb_var = to_variable(text_emb)
				title_length_var = to_variable(torch.from_numpy(title_lengths).float())
				text_length_var = to_variable(torch.from_numpy(text_lengths).float())

				cos_sim = model(title_emb_var, title_length_var, title_rev, text_emb_var, text_length_var, text_rev, 1, dev=True)
				cos_arr = cos_sim.cpu().data.numpy()[0]
				true_labels = dev_data[dev_i][2]
				pred_tmp = [x for _,x in sorted(zip(cos_arr, true_labels), reverse=True)]
				dev_pred.append(pred_tmp)

			metric = Evaluation(dev_pred)
			printf('Dev metric:')
			printf('MAP:', metric.MAP())
			printf('MRR:', metric.MRR())
			printf('P@1:', metric.Precision(1))
			printf('P@5:', metric.Precision(5))
			# print('Dev done')
		
		del cos_sim
		del const_y
		del title_emb_var
		del text_emb_var
		del title_length_var
		del text_length_var
		del text_emb, text_lengths, text_rev
		del title_emb, title_lengths, title_rev
		torch.cuda.empty_cache()
	fn = os.path.join(result_folder, "model_epoch_{}".format(n_epoch+1))
	if save_to_file:
		torch.save({
			"model": model.state_dict(),
			"opt": optimizer.state_dict()
		}, fn+"_model")