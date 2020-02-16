import json
import argparse
from os.path import isfile, join
import re
import numpy as np
import pprint
import pickle
import tensorflow as tf
import utils
import vis_lstm_model
import time
import h5py
from keras.utils.vis_utils import plot_model
import scikitplot as skplt
import matplotlib.pyplot as plt



def prepare_training_data(trainquestionfile,trainannotations,valquestionfile,valannotations, data_dir='Data'):
        t_q_json_file = join(data_dir, 'trainquestions.json')
        t_a_json_file = join(data_dir, 'trainannotations.json')
        v_q_json_file = join(data_dir,'valquestions.json')
        v_a_json_file = join(data_dir,'valannotations.json')
        qa_data_file = join(data_dir, 'newqadata.pkl')
        vocab_file = join(data_dir, 'newvocab.pkl')
        if isfile(qa_data_file):
            with open(qa_data_file) as f:
                data = pickle.load(f)
                return data
        print ("Loading Training questions")
        with open(t_q_json_file) as f:
            t_questions = json.loads(f.read())
        print ("Loading Training anwers")
        with open(t_a_json_file) as f:
            t_answers = json.loads(f.read())
        print ("Loading Val questions")
        with open(v_q_json_file) as f:
            v_questions = json.loads(f.read())
        print ("Loading Val answers")
        with open(v_a_json_file) as f:
            v_answers = json.loads(f.read())
        print ("Ans", len(t_answers['annotations']), len(v_answers['annotations']))
        print ("Qu", len(t_questions['questions']), len(v_questions['questions']))
        answers = t_answers['annotations'] + v_answers['annotations']
        questions = t_questions['questions'] + v_questions['questions']
        answer_vocab = make_answer_vocab(answers)
        question_vocab, max_question_length = make_questions_vocab(questions, answers, answer_vocab)
        print ("Max Question Length", max_question_length)
        word_regex = re.compile(r'\w+')
        training_data = []
        for i,question in enumerate( t_questions['questions']):
            ans = t_answers['annotations'][i]['multiple_choice_answer']
            if ans in answer_vocab:
                training_data.append({
				'image_id' : t_answers['annotations'][i]['image_id'],
				'question' : np.zeros(max_question_length),
				'answer' : answer_vocab[ans]
				})
                question_words = re.findall(word_regex, question['question'])
                base = max_question_length - len(question_words)
                for i in range(0, len(question_words)):
                    training_data[-1]['question'][base + i] = question_vocab[ question_words[i] ]
        print( "Training Data", len(training_data))
        val_data = []
        for i,question in enumerate( v_questions['questions']):
            ans = v_answers['annotations'][i]['multiple_choice_answer']
            if ans in answer_vocab:
                val_data.append({
				'image_id' : v_answers['annotations'][i]['image_id'],
				'question' : np.zeros(max_question_length),
				'answer' : answer_vocab[ans]
				})
                question_words = re.findall(word_regex, question['question'])
                base = max_question_length - len(question_words)
                for i in range(0, len(question_words)):
                    val_data[-1]['question'][base + i] = question_vocab[ question_words[i] ]
        print ("Validation Data", len(val_data))
        data = {
		'training' : training_data,
		'validation' : val_data,
		'answer_vocab' : answer_vocab,
		'question_vocab' : question_vocab,
		'max_question_length' : max_question_length
	}
        print ("Saving qa_data")
        with open(qa_data_file, 'wb') as f:
            pickle.dump(data, f)
        with open(vocab_file, 'wb') as f:
            vocab_data = {
			'answer_vocab' : data['answer_vocab'],
			'question_vocab' : data['question_vocab'],
			'max_question_length' : data['max_question_length']
		}
            pickle.dump(vocab_data, f)
        return data
	
def load_questions_answers(filename='newqadata.pkl', data_dir = 'Data'):
	qa_data_file = join(data_dir, filename)
	
	if isfile(qa_data_file):
		with open(qa_data_file,'rb') as f:
			data = pickle.load(f)
			return data

def get_question_answer_vocab(filename='newvocab.pkl', data_dir = 'Data'):
	vocab_file = join(data_dir, filename)
	vocab_data = pickle.load(open(vocab_file,'rb'))
	return vocab_data

def make_answer_vocab(answers):
	top_n = 1000
	answer_frequency = {} 
	for annotation in answers:
		answer = annotation['multiple_choice_answer']
		if answer in answer_frequency:
			answer_frequency[answer] += 1
		else:
			answer_frequency[answer] = 1

	answer_frequency_tuples = [ (-frequency, answer) for answer, frequency in answer_frequency.items()]
	answer_frequency_tuples.sort()
	answer_frequency_tuples = answer_frequency_tuples[0:top_n-1]

	answer_vocab = {}
	for i, ans_freq in enumerate(answer_frequency_tuples):
		# print i, ans_freq
		ans = ans_freq[1]
		answer_vocab[ans] = i

	answer_vocab['UNK'] = top_n - 1
	return answer_vocab


def make_questions_vocab(questions, answers, answer_vocab):
	word_regex = re.compile(r'\w+')
	question_frequency = {}

	max_question_length = 0
	for i,question in enumerate(questions):
		ans = answers[i]['multiple_choice_answer']
		count = 0
		if ans in answer_vocab:
			question_words = re.findall(word_regex, question['question'])
			for qw in question_words:
				if qw in question_frequency:
					question_frequency[qw] += 1
				else:
					question_frequency[qw] = 1
				count += 1
		if count > max_question_length:
			max_question_length = count


	qw_freq_threhold = 0
	qw_tuples = [ (-frequency, qw) for qw, frequency in question_frequency.items()]
	# qw_tuples.sort()

	qw_vocab = {}
	for i, qw_freq in enumerate(qw_tuples):
		frequency = -qw_freq[0]
		qw = qw_freq[1]
		# print frequency, qw
		if frequency > qw_freq_threhold:
			# +1 for accounting the zero padding for batc training
			qw_vocab[qw] = i + 1
		else:
			break

	qw_vocab['UNK'] = len(qw_vocab) + 1

	return qw_vocab, max_question_length


def load_fc7_features(data_dir, split):
	import h5py
	fc7_features = None
	image_id_list = None
	with h5py.File( join( data_dir, ('fc7new.h5')),'r') as hf:
		fc7_features = np.array(hf.get('fc7_features'))
	with h5py.File( join( data_dir, ('image_id_listnew.h5')),'r') as hf:
		image_id_list = np.array(hf.get('image_id_list'))
	return fc7_features, image_id_list



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_lstm_layers', type=int, default=2,
                       help='num_lstm_layers')
    parser.add_argument('--fc7_feature_length', type=int, default=4096,
                       help='fc7_feature_length')
    parser.add_argument('--rnn_size', type=int, default=512,
                       help='rnn_size')
    parser.add_argument('--embedding_size', type=int, default=512,
                       help='embedding_size'),
    parser.add_argument('--word_emb_dropout', type=float, default=0.5,
                       help='word_emb_dropout')
    parser.add_argument('--image_dropout', type=float, default=0.5,
                       help='image_dropout')
    parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
    parser.add_argument('--batch_size', type=int, default=200,
                       help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Batch Size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Epochs')
    parser.add_argument('--debug', type=bool, default=False,
                       help='Debug')
    parser.add_argument('--resume_model', type=str, default=None,
                       help='Trained Model Path')
    parser.add_argument('--version', type=int, default=2,
                       help='VQA data version')

    args = parser.parse_args()    
    print("Creating QuestionAnswer data")
    prepare_training_data('trainquestions.json','trainannotations.json','valquestions.json','valannotations.json')
    print("Prepared given data")
    print("Reading QuestionAnswer data")

    qa_data=load_questions_answers('newqadata.pkl','Data')
    print(qa_data['answer_vocab'])

    print("Creating Image features")
################################################
    split='train'
    vgg_file = open('Data/vgg16.tfmodel','rb')
    vgg16raw = vgg_file.read()
    vgg_file.close()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(vgg16raw)

    images = tf.placeholder("float", [None, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={ "images": images })

    graph = tf.get_default_graph()

    for opn in graph.get_operations():
        print ("Name", opn.name, opn.values())

    all_data = load_questions_answers()
    if split == "train":
	    qa_data = all_data['training']
    else:
	    qa_data = all_data['validation']
	
    image_ids = {}
    for qa in qa_data:
	    image_ids[qa['image_id']] = 1

    image_id_list = [img_id for img_id in image_ids]
    print ("Total Images", len(image_id_list))
	
	
    sess = tf.Session()
    fc7 = np.ndarray( (len(image_id_list), 4096 ) )
    idx = 0

    while idx < len(image_id_list):
	    start = time.clock()
	    image_batch = np.ndarray( (10, 224, 224, 3 ) )

	    count = 0
	    for i in range(0, args.batch_size):
		    if idx >= len(image_id_list):
			    break
		    image_file = join('Data', '%snew/%.1d.jpg'%(split, image_id_list[idx]) )
		    image_batch[i,:,:,:] = utils.load_image_array(image_file)
		    idx += 1
		    count += 1
		
		
	    feed_dict  = { images : image_batch[0:count,:,:,:] }
	    fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
	    fc7_batch = sess.run(fc7_tensor, feed_dict = feed_dict)
	    fc7[(idx - count):idx, :] = fc7_batch[0:count,:]
	    end = time.clock()
	    print ("Time for batch 10 photos", end - start)
	    print ("Hours For Whole Dataset" , (len(image_id_list) * 1.0)*(end - start)/60.0/60.0/10.0)

	    print ("Images Processed", idx)

		

    print ("Saving fc7 features")
    h5f_fc7 = h5py.File( join('Data','fc7new.h5'), 'w')
    h5f_fc7.create_dataset('fc7_features', data=fc7)
    h5f_fc7.close()

    print ("Saving image id list")
    h5f_image_id_list = h5py.File( join('Data','image_id_listnew.h5'), 'w')
    h5f_image_id_list.create_dataset('image_id_list', data=image_id_list)
    h5f_image_id_list.close()
    print ("Done!")
    

    
    ##################################################33
    print("Reading image features")
    fc7_features,image_id_list=load_fc7_features('Data','train')
    print ("FC7 features", fc7_features.shape)
    print ("image_id_list", image_id_list.shape)
    qa_data=load_questions_answers('newqadata.pkl','Data')
    print(qa_data['answer_vocab'])
    image_id_map = {}
    for i in range(len(image_id_list)):
        image_id_map[ image_id_list[i] ] = i
    ans_map = { qa_data['answer_vocab'][ans] : ans for ans in qa_data['answer_vocab']}
    model_options = {
		'num_lstm_layers' : 2,
		'rnn_size' : 512,
		'embedding_size' : 512,
		'word_emb_dropout' : 0.5,
		'image_dropout' : 0.5,
		'fc7_feature_length' : 4096,
		'lstm_steps' : qa_data['max_question_length'] + 1,
		'q_vocab_size' : len(qa_data['question_vocab']),
		'ans_vocab_size' : len(qa_data['answer_vocab'])
	}
    model = vis_lstm_model.Vis_lstm_model(model_options)
    input_tensors, t_loss, t_accuracy, t_p = model.build_model()
    train_op = tf.train.AdamOptimizer(0.001).minimize(t_loss)
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    #model.summary()
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    if args.resume_model:
        saver.restore(sess, args.resume_model)
    for i in range(100):
        batch_no = 0
        while (batch_no*10) < len(qa_data['training']):
            sentence, answer, fc7 = get_training_batch(batch_no, 10, fc7_features, image_id_map, qa_data, 'train')
            _, loss_value, accuracy, pred = sess.run([train_op, t_loss, t_accuracy, t_p],
                                                     feed_dict={
                                                         input_tensors['fc7']:fc7,
                                                         input_tensors['sentence']:sentence,
                                                         input_tensors['answer']:answer
                                                         }
                                                     )
            batch_no += 1
            if args.debug:
                for idx, p in enumerate(pred):
                    print (ans_map[p], ans_map[ np.argmax(answer[idx])])
                print ("Loss", loss_value, batch_no, i)
                print ("Accuracy", accuracy)
                print ("---------------")
                skplt.metrics.plot_roc_curve(answer[idx], ans_map[p])
                plt.show()
            else:
                print ("Loss", loss_value, batch_no, i)
                print ("Training Accuracy", accuracy)
                #skplt.metrics.plot_roc_curve(answer[0], pred[0])
                #plt.show()
        save_path = saver.save(sess, "Data/Models/modelnew{}.ckpt".format(i))

def get_training_batch(batch_no, batch_size, fc7_features, image_id_map, qa_data, split):
	qa = None
	if split == 'train':
		qa = qa_data['training']
	else:
		qa = qa_data['validation']

	si = (batch_no * batch_size)%len(qa)
	ei = min(len(qa), si + batch_size)
	n = ei - si
	sentence = np.ndarray( (n, qa_data['max_question_length']), dtype = 'int32')
	answer = np.zeros( (n, len(qa_data['answer_vocab'])))
	fc7 = np.ndarray( (n,4096) )

	count = 0
	for i in range(si, ei):
		sentence[count,:] = qa[i]['question'][:]
		answer[count, qa[i]['answer']] = 1.0
		fc7_index = image_id_map[ qa[i]['image_id'] ]
		fc7[count,:] = fc7_features[fc7_index][:]
		count += 1
	
	return sentence, answer, fc7

if __name__ == '__main__':
	main()

