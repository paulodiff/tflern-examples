'''
Boston House TF-Learn

Tutorial for test Boston House

Dati : housing.csv

'''

import numpy as np
import tflearn
import argparse
import pandas
import random

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.metrics import R2
from statistics import median, mean
from collections import Counter
from tabulate import tabulate
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error



def initial_population(training_input_file_name, training_data_percentage, save_data_file_header):
    print('#initial_population ...')
    # print('initial_games:',training_data_file_name)
    print('training_input_file_name:',training_input_file_name )
    print('training_data_percentage:',training_data_percentage )
    print('save_data_file_header:',save_data_file_header )
    
    
    dataframe = pandas.read_csv(training_input_file_name, delim_whitespace=True, header=None)
    dataset = dataframe.values
        
    
    dataset_size = len(dataset)
    training_size = int(dataset_size * training_data_percentage / 100)
    test_size = dataset_size - training_size
    
    print('Dataset size:',dataset_size)
    print('Training size:',training_size)
    print('Test size:',test_size)
    
    
    print('Shuffle ....')
    random.shuffle(dataset)
    
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    print('MinMaxScaler ....')
    dataset_minmax = min_max_scaler.fit_transform(dataset)
    
    
    training_data = dataset_minmax[:training_size]
    test_data = dataset_minmax[training_size :dataset_size,]
    evaluate_data = []
    
    print('-- training_data --')
    print(tabulate(training_data))
    
    print('-- test_data --')
    print(tabulate(test_data))
    
    print('-- evaluate_data --')
    print(tabulate(evaluate_data))
        
    # Dataset_minmax = min_max_scaler.fit_transform(dataset)

    
    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    training_file_name = save_data_file_header + '_TRAINING_DATA.npy'
    np.save(training_file_name, training_data_save)
    print(training_file_name, ' saved!')
    print(len(training_data_save), '  items ')

    test_data_save = np.array(test_data)
    test_file_name = save_data_file_header + '_TEST_DATA.npy'
    np.save(test_file_name, test_data_save)
    print(test_file_name, ' saved!')
    print(len(test_data_save), '  items ')

    # evaluate_data_save = np.array(evaluate_data)
    # np.save('EVALUATE_DATA.npy', evaluate_data_save)

    
    # some stats here, to further illustrate the neural network magic!
    # print('training_data_save:', len(training_data_save))
    # print(training_data_save)
    # print('Average accepted score:',mean(accepted_scores))
    # print('Median score for accepted scores:',median(accepted_scores))
    # print(Counter(accepted_scores))
    # print(Counter(training_data_save))
    # print(accepted_scores)

    # X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    # y = [i[1] for i in training_data]

    
    # return training_data, test_data, y, len(X[0])
    return training_data, test_data

'''
Caricamento dei dati
Preparazione set dei dati di test


traininig_or_test : TRAINING or TEST

'''
    
def load_data(save_data_file_header, net_input_size, traininig_or_test):
    
    log_info = '#load_training_data'
    
    file_to_load = save_data_file_header + '_' +  traininig_or_test + '_DATA.npy'
    print(log_info)
    print(log_info, 'file_to_load: ', file_to_load)
    print(log_info, 'net_input_size: ', net_input_size)
    
    training_data_save = np.load(file_to_load)
    training_data = training_data_save.tolist()
    
    
    # some stats here, to further illustrate the neural network magic!
    print(log_info, 'training_data_save:', len(training_data_save))

    # X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    # y = [i[1] for i in training_data]
    
    int_size = int(net_input_size)
    
    X = (training_data_save[:,0:int(net_input_size)]).reshape(-1,net_input_size)
    y = (training_data_save[:,int(net_input_size)]).reshape([-1, 1])
    
    print(log_info,'X shape: ', X.shape)
    #print(X) print(X.shape) print(len(X))
    print(log_info,'len(X) : ', len(X))
    #input("Press Enter to continue...")
    
    print(log_info, 'y shape', y.shape)
    print(log_info,'len(y) : ',len(y))
    
    #print(y2)
    #input("Press Enter to continue...")
       
    # print(y.shape)
    # input_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    # y_true = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    # X = np.array([1,-1,1,1,-1,-1]).reshape([-1, 1])
    # Y = np.array([1,0,1,0,0,1]).reshape([-1, 1])
    
    return training_data, X, y


# definizione della rete 

def neural_network_model(input_size, learning_rate):
    print('#neural_network_model')
    print('input_size:',input_size)
    print('learning_rate:',learning_rate)
    
    
    ''' 
    network = input_data(shape=[None, input_size], name='input')

    network = fully_connected(network, 2, activation='relu')
    network = dropout(network, 0.8)

    #### network = fully_connected(network, 512, activation='relu')
    ##### network = dropout(network, 0.8)
    
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')
    '''
    
    network = input_data(shape=[None, input_size], name='input')

    #network = fully_connected(network, 13, activation='relu')
    # network = dropout(network, 0.8)

    network = fully_connected(network, 32, activation='relu')
    network = dropout(network, 0.8)

    #network = fully_connected(network, 64, activation='relu')
    #network = dropout(network, 0.8)
    
    network = fully_connected(network, 1, activation='linear')
    # network = regression(network, optimizer='adam', learning_rate=learning_rate, loss='mean_square', name='targets')
    #r2 = R2()
    network = regression(network, optimizer='sgd', learning_rate=0.01, loss='mean_square', name='targets', metric='R2')
    
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(model, X, y, X_test, y_test, model_file_name):

    log_info = '#train_model:'

    print(log_info)

    # X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    # y = [i[1] for i in training_data]
    
    
    #print('----------------------------------------------------------------------')
    #print('input_size:',len(X))
    #print(X)
    #input("Press Enter to continue evaluation...")
    #print(y)
    #input("Press Enter to continue evaluation...")
    # input_size_network = len(X[0])
    
    
    #if not model:
    #    print('#train_model#build_model')
    #    model = neural_network_model(input_size = len(X[0]), learning_rate)
    
    print(log_info,'training')
    model.fit(  {'input': X}, {'targets': y}, 
                validation_set=({'input': X_test}, {'targets': y_test}),
                n_epoch=1000, batch_size=10, show_metric=True, snapshot_epoch=False)
                
    '''
    model.fit({'input': X}, 
              {'targets': y}, 
              n_epoch=10, 
              #snapshot_step=500, 
              batch_size=10,
              show_metric=True, 
              validation_set=0.2,
              run_id='house_learning')
    '''
    print(log_info,'saving:', model_file_name)
    
    # model.save(model_file_name)
    
    print('----------------------------------------------------------------------')
    
    score = model.evaluate(X, y)
    print('Train accuracy: %0.4f%%' % (score[0] * 100))
    
    score = model.evaluate(X_test, y_test)
    print('Test accuracy: %0.4f%%' % (score[0] * 100))

    
    y_prediction = model.predict(X_test)
    # print(y_prediction)
    print('MSE X_test: %0.4f' % ( mean_squared_error(y_test,y_prediction)) )
    
    
    return model
    

def evaluate_model(model, X, y):    
    print('#evaluate_model')
    # Evaluate model
    
    score = model.evaluate(X, y)
    print(score)
    print('Test accuracy: %0.4f%%' % (score[0] * 100))
    
    
    y_prediction = model.predict(X)
    print(y_prediction)
    print('MSE Test: %.3f' % ( mean_squared_error(y,y_prediction)) )
    print(y)
    print(X)

    # Run the model on one example
    # prediction = model.predict([test_x[0]])
    # print("Prediction: %s" % str(prediction[0]))
    

    return
    
def predict_model(model, goal_steps):

    print('#predict_model')

    scores = []
    choices = []
    for this_game in range(10):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
    
        print('########################################################################################')
        print('########################################################################################')
        print('########################################################################################')
    
        for cur_step in range(goal_steps):
            env.render()

            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                print(this_game, '-', cur_step, ') prev_obs:',prev_obs)
                prev_obs_reshaped = prev_obs.reshape(-1,len(prev_obs),1)
                # print(prev_obs_reshaped)
                prediction = model.predict(prev_obs.reshape(-1,len(prev_obs),1))
                print(prediction)
                action = np.argmax(prediction[0])
                # print(action)
                # action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

            choices.append(action)
                    
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            if done: break

        scores.append(score)

    print('Average Score:',sum(scores)/len(scores))
    print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
    print(scores)


# @Gooey()
def main():
    parser = argparse.ArgumentParser(description='Script to train CartPole')
    parser.add_argument("--model_file_name", 
                                default='boston-house.tflearn',
                                help="file to read/save model")
    parser.add_argument("--action", 
                                default="generate",
                                choices=['train', 'evaluate', 'generate'],
                                # default="mobile_ssd/ssd_mobilenet_v1_coco.pbtxt",
                                help='train or evaluate on generate')
    parser.add_argument("--net_input_size", 
                                default=13,
                                type=int,
                                # default="mobile_ssd/ssd_mobilenet_v1_coco.pbtxt",
                                help='network input size')                                
    parser.add_argument("--goal_steps", 
                                default=1000,
                                type=int,
                                help='Numero di partite da effetture')
    parser.add_argument("--score_requirement", 
                                default=60,
                                type=int,
                                help='Punteggio minimo per selezionare i dati di una partita casuale')
    parser.add_argument("--initial_games", 
                                default=20000,
                                type=int,
                                help='Numero di partite da effettuare per la generazione dei dati di training')
    parser.add_argument("--learning_rate", 
                                default=1e-2,
                                type=float,
                                help='Learning rate')
    parser.add_argument("--training_data_file_name", 
                                default="boston-house-training-data.npy",
                                help='File name data training')
    parser.add_argument("--input_data_file_name", 
                                default="housing.csv",
                                help='File name input data')
    parser.add_argument("--save_data_file_header", 
                                default="HOUSING",
                                help='Header File name dave data')
    parser.add_argument("--training_data_percentage", 
                                default=80.0,
                                type=float,
                                help='Data input percentage')
                                


    
    args = parser.parse_args()
    print(args)
    print('Start ...')
    print('args.action:', args.action)
    print('args.model_file_name:', args.model_file_name)
    print('args.net_input_size', args.net_input_size)
    print('args.training_data_file_name', args.training_data_file_name)
    
    # print('args.num_classes:', args.num_classes)
    
    
    if args.action=='train':
        print(args.action)
        # training_data, X, y, net_input_size = initial_population(args.initial_games, args.goal_steps, args.score_requirement, args.training_data_file_name)   
        # input("Press Enter to continue...")
        training_data, X, y = load_data(args.save_data_file_header, args.net_input_size, 'TRAINING')
        training_data, X_test, y_test = load_data(args.save_data_file_header, args.net_input_size, 'TEST')
        model = neural_network_model( args.net_input_size, args.learning_rate)        
        # exit(1)
        input("Press Enter to train!...")
        train_model(model, X, y, X_test, y_test, args.model_file_name)    

        
    elif args.action == 'evaluate':
        print(args.action)
        print(args.action, 'model_file_name:', args.model_file_name)
        print(args.action, 'goal_steps:', args.goal_steps)
        training_data, X, y = load_data(args.save_data_file_header, args.net_input_size, 'TEST')
        model = neural_network_model( args.net_input_size, args.learning_rate)      
        print(args.action, 'model load:', args.model_file_name)
        model.load(args.model_file_name)
        input("Press Enter to continue evaluation...")
        evaluate_model(model, X, y)
        
    elif args.action == 'generate':
        print(args.action)
        training_data, test_data = initial_population(args.input_data_file_name, args.training_data_percentage, args.save_data_file_header)   
        # model = neural_network_model( args.net_input_size, args.learning_rate)        
        # model.load(args.model_file_name)
        # evaluate_model(model, args.goal_steps)
    
    else:
        print('# NO VALID action')

        
if __name__ == '__main__':
  main()        