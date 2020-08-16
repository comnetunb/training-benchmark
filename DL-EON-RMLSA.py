"""
Created on Tue JUN 10  2020

RSA com Modulacoes BPSK QPSK 8QAM
                                        Colocando USANet e PanEURO

@author: guilhermeeneas
"""

from keras.layers import Conv1D, MaxPooling1D, Flatten  # CNN 1D
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import shuffle
import random
from definitions import Topology
from configuration import use_gpu, use_cnn, seed_count, epoch_count, topology, algorithms, loads
# camadas densas - todos neuronios sao ligados e tal...
from keras.layers import Dense, Dropout
# RN do tipo sequencial: entrada - camadas - saida
from keras.models import Sequential
from keras.utils import np_utils
import keras
import numpy as np
import pandas as pd
import time
inicio = time.time()
# import matplotlib.pyplot as plt  #botei agora pra testar grafico
#import gui_functions as gui


"""
################################################################################################
                                        Arrumando Parametros
################################################################################################
"""

# porcentagem_de_features = 1   # default eh 40

quantidade_das_amostras_de_treinamento = 10080 * seed_count
batch_size = quantidade_das_amostras_de_treinamento // 1
# OBS.: em 1 semente o len(filenames_treinamento) eh 10080, em 2 eh 20160, em 3 eh 30240, em 4 eh 40320, em 5 eh 50400


"""
################################################################################################
                                        BUFFER - Teste
################################################################################################
"""

numero_de_algoritmos = len([x for x in algorithms() if x != ''])

filenames = []
contador_sementes = 0

nomes_algoritmos = algorithms()
# Criar nomes dos arquivos
for semente in ['10', '1', '7', '16', '22']:
    contador_sementes = contador_sementes + 1
    for index in range(numero_de_algoritmos):
        for carga in loads():
            for i in range(10, 100, 1):
                nome_arquivo = semente + '_' + \
                    nomes_algoritmos[index] + '_' + carga + \
                    '_state_' + str(i+1) + '000.txt'
                filenames.append(nome_arquivo)
    if seed_count == contador_sementes:
        break


numero_de_estados_por_carga = 90  # de 11.000 a 100.000
numero_de_cargas = len(loads())
inputs_por_algoritmo = numero_de_estados_por_carga * \
    numero_de_cargas * seed_count  # por algoritmo


# Criando as classes
inputs_total = numero_de_algoritmos * inputs_por_algoritmo
classes = np.zeros((inputs_total, 1))
count = 0
for i in range(numero_de_algoritmos):
    for j in range(inputs_por_algoritmo):
        classes[count, 0] = i                 # [0,0,0,..,1,1,1,...,2,2,2,...]
        count += 1
        # resposta.append(i) --> poderia ter feito resposta uma lista e depois ter criado um pandas, mas vou deixar assim
# classe = pd.DataFrame(resposta) -->  antes fazia desse tipo... agora mudei


''' ### dando um shuffle ###  '''
filenames, classes = shuffle(filenames, classes)

# one hot encoder (antiga classes_dummy)
classes_one_hot = np_utils.to_categorical(classes)


if topology == Topology.USANET:
    num_links = 86
    numero_features = 27520
elif topology == Topology.PANEURO:
    num_links = 82
    numero_features = 26240


# Organizando em previsores e classes
#previsores = pd.DataFrame(entrada)


"""
################################################################################################
                                        Pre - Processamento
################################################################################################
"""
'''
num_estados = 1   #apenas 1 estado... nao mudar

nomes_algoritmos = algorithms()
numero_de_algoritmos = 0  
for i in nomes_algoritmos:
    if i != '':
        numero_de_algoritmos += 1

algoritmo1, algoritmo2, algoritmo3, algoritmo4, algoritmo5, algoritmo6, algoritmo7,algoritmo8,algoritmo9,algoritmo10 = gui.criar_e_ler_algoritmos(num_sementes,num_estados,numero_de_algoritmos,nomes_algoritmos,topology)  #nao pego o numero de algoritmos pq eu crio 5 vetores de qlq forma
#SP,SPV,CS,K3SP,SP_Random,x,y  = gui.criar_algoritmos(num_sementes,num_estados) #passo o numero de sementes como parametro

tempo_leitura = time.time()
print('\nTempo de leitura das sementes: ',round((tempo_leitura-inicio)/60),' minutos')

entrada, resposta = gui.criar_entrada_e_resposta(algoritmo1, algoritmo2, algoritmo3, algoritmo4, algoritmo5, algoritmo6, algoritmo7, algoritmo8,algoritmo9, algoritmo10, num_sementes, numero_de_algoritmos)

'''


'''
######################################################################################################
                                 #  SELECAO DE FEATURES    
                              nao da pra ter com buffer...
######################################################################################################
'''

'''
from sklearn.feature_selection import chi2, SelectPercentile
     
porcentagem_features = porcentagem_de_features   #40
selector = SelectPercentile(chi2, percentile = porcentagem_features) 
previsores = selector.fit_transform(previsores, classe_dummy)

#olhar essas duas variaveis pra entender quais as colunas (features) foram selecionadas
features_indices = selector.get_support(indices=True)
features_booleano = selector.get_support(indices=False)

#criando arquivo para salvar features
df_features_booleano = pd.DataFrame(features_booleano, columns=['Features']) 
if topology == Topology.USANET:
    df_features_booleano.to_csv('features_booleano_RMLSA_usanet.csv',index=False) 
elif topology == Topology.PANEURO:
    df_features_booleano.to_csv('features_booleano_RMLSA_paneuro.csv',index=False) 


numero_features = previsores.shape[1]

#Lembrar: esse features_booleano eu boto num arquivo pra depois usa-lo na identificacao das melhores features

#fim teste de selecao de features
'''


'''
######################################################################################################
                                #     GPU STUFF     
######################################################################################################
'''

if use_gpu:
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


'''
######################################################################################################
                                #   CROSS VALIDATION     
######################################################################################################
'''

# vamos fazer a separacao entre treinamento e teste agora
# antigamente usava isso aqui abaixo... mas aora eu criei meu proprio metodo de separacao
#from sklearn.model_selection import train_test_split
#previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)
''' ### lembrando:   tamanho_entrada agora eh inputs_total ### '''
# exemplo: 2 algoritmos com 1 semente: 2520
k = 5
seed = 7  # o seed eh pra garantir a reprodutibilidade ... aqui eh do random

# metodo mais novinho pra dividir meus indices em folds (para cross-validation)


# o seed eh pra garantir a reprodutibilidade
def kfold_dividindo_indices(indices, k, seed=7):
    size = len(indices)
    subset_size = round(size / k)  # tamanho do subset
    subset_size = int(subset_size)
    random.Random(seed).shuffle(indices)  # aleatorizando os indices
    # criando uma nova lista em que cada elemento eh um dos subsets criados
    subsets = [indices[x:x+subset_size]
               for x in range(0, len(indices), subset_size)]
    kfolds = []
    for i in range(k):  # agora criando os k folds, ou seja, k conjuntos com a subdivisao treino/teste
        test = subsets[i]
        train = []
        for subset in subsets:
            if subset != test:
                train.append(subset)
        kfolds.append((train, test))
    return kfolds


# o metodo anterior divide os indices. Aqui eu divido o dataset de fato em folds!
def treino_teste_split(previsores, classe_dummy, inputs_total, k, seed, fold):
    # gerando aqui uma lista com numeros de 0 a 6299
    indices = list(range(inputs_total))
    # resposta eh uma lista com 5 tuplas
    resposta = kfold_dividindo_indices(indices, k, seed)

    # sao 5 folds, logo tenho 5 tuplas. Pego cada uma delas:
    fold1, fold2, fold3, fold4, fold5 = resposta

    if fold == 1:  # notoriamente isso nao ta generalizado! Botei so pra 5 folds por enquanto hehe
        fold = fold1
    elif fold == 2:
        fold = fold2
    elif fold == 3:
        fold = fold3
    elif fold == 4:
        fold = fold4
    else:
        fold = fold5

    previsores_treino_fold = []
    previsores_teste_fold = []
    for i in fold[1]:  # fold eh uma tupla. O elemento 0 sao os 4 grupos de treinamentos e o elemento 1 eh o grupo de teste. Cada grupo tem os indices
        # aqui coloco os indices de teste do fold1
        previsores_teste_fold.append(previsores[i])
    # percorro cada um dos grupos (com k=5 tenho 4 grupos de treinamento que serao agrupados em 1 so)
    for lista in fold[0]:
        for indice in lista:
            # ai aqui junto os de treinamento tudim numa so lista
            previsores_treino_fold.append(previsores[indice])

    # transformando em array numpy pra ficar igual a previsores
    previsores_teste_fold = np.asarray(previsores_teste_fold)
    previsores_treino_fold = np.asarray(previsores_treino_fold)

    # de maneira similar, vamos pegar as classes_dummy de treino e teste de acordo com os indices selecionados nesse fold
    classe_dummy_treino_fold = []
    classe_dummy_teste_fold = []
    for i in fold[1]:
        classe_dummy_teste_fold.append(classe_dummy[i])
    for lista in fold[0]:
        for indice in lista:
            classe_dummy_treino_fold.append(classe_dummy[indice])

    # transformando em array numpy pra ficar igual a previsores
    classe_dummy_teste_fold = np.asarray(classe_dummy_teste_fold)
    classe_dummy_treino_fold = np.asarray(classe_dummy_treino_fold)

    return previsores_treino_fold, previsores_teste_fold, classe_dummy_treino_fold, classe_dummy_teste_fold


'''
######################################################################################################
                            Criacao do Gerador Personalizado 
Custom Generator: As our dataset is too large to fit in memory, we have to load the dataset 
from the hard disk in batches to our memory.
Link: https://github.com/mrrajatgarg/kaggle/blob/master/Training_On_Large_Dataset_Final.ipynb
######################################################################################################
'''


class My_Custom_Generator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):   # aqui eh o construtor
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):  # calcula o numero de batches que o gerador vai produzir
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx *
                                       self.batch_size: (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx+1) * self.batch_size]

        return (np.array([
                resize(imread('/content/all_images/' +
                              str(file_name)), (80, 80, 3))
                for file_name in batch_x])/255), np.array(batch_y)

        '''
######################################################################################################
                            Meu Gerador
Custom Generator: As our dataset is too large to fit in memory, we have to load the dataset 
from the hard disk in batches to our memory.
Segundo a documentação, o gerador tem que gerar uma tupla (inputs, targets)  
Link:   https://faroit.com/keras-docs/1.2.2/models/sequential/  
        https://keras.io/models/sequential/   
Outros links importantes na criacao do meu gerador:
        https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
        https://www.programmersought.com/article/52471223775/          --> olhar o predict tbm  

Ler tbm se tiver tempo:        
        https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
        
######################################################################################################
'''


def gerador(filename, classe, tamanho_do_batch, numero_features):
    batch_previsores = np.zeros((len(filename), numero_features), dtype=int)
    # numero de colunas eh o numero de algoritmos... ideia do one hot encoder
    batch_classes = np.zeros((len(filename), classe.shape[1]), dtype=int)
    #batch_previsores = np.zeros((tamanho_do_batch,numero_features), dtype=int)
    # batch_classes = np.zeros((tamanho_do_batch,classe.shape[1]), dtype=int) #numero de colunas eh o numero de algoritmos... ideia do one hot encoder

    i = 0  # indexador da posicao da entrada no batch de entradas
    while True:  # MUDAAAAAAAAAAAAAAAAAAAAAAAR
        #print('Entrei no while')
        for nome_arquivo in filename:
            # print(i)
            data = pd.read_csv(nome_arquivo, delimiter=';')
            # pre-processamento:
            data = data.drop(['Unnamed: 320'], axis=1)
            # data.shape[0] eh o mesmo que num_link...
            entrada = data[0:num_links].values
            # transformo agora em 1 dimensao so: (1,27520) para usanet e (1,26240) para paneuro
            entrada = entrada.reshape(1, -1)
            '''  Colocando so 0 e 1  '''
            for j in range(0, numero_features, 1):   # 27520 ou 26240
                if entrada[0][j] != 0:
                    entrada[0][j] = 1
            batch_previsores[i] = entrada
            batch_classes[i] = classe[i]
            # if (i+1) % tamanho_do_batch == 0:
            #    print("i = ", i,", multiplo = ",i+1)
            #    #break
            # produz valores e joga tipo como retorno da funcao
            yield batch_previsores, batch_classes
            i += 1
            # print(i)
        # return batch_previsores,batch_classes     # tirei agora


def fase_de_testes_ou_validacao(filenames_teste, numero_features):
    # fazendo especificamente a parte de TESTE / validacao
    previsores_teste = np.zeros(
        (len(filenames_teste), numero_features), dtype=int)
    i = 0
    for nome_arquivo in filenames_teste:
        data = pd.read_csv(nome_arquivo, delimiter=';')
        # pre-processamento:
        data = data.drop(['Unnamed: 320'], axis=1)
        # data.shape[0] eh o mesmo que num_link...
        entrada = data[0:num_links].values
        # transformo agora em 1 dimensao so: (1,27520) para usanet e (1,26240) para paneuro
        entrada = entrada.reshape(1, -1)
        '''  Colocando so 0 e 1  '''
        for j in range(0, numero_features, 1):   # 27520 ou 26240
            if entrada[0][j] != 0:
                entrada[0][j] = 1
        previsores_teste[i] = entrada
        i += 1
    return previsores_teste


'''
######################################################################################################
                             Acabaram os metodos do Cross Validation    
                               Agora preparar para o treinamento    
######################################################################################################
'''


# Used this line as our filename array is not a numpy array.
filenames = np.array(filenames)

for fold in [1]:  # [1,2,3,4,5]:

    filenames_treinamento, filenames_teste, classe_treinamento, classe_teste = treino_teste_split(
        filenames, classes_one_hot, inputs_total, k, seed, fold)

    dropout = 0.1  # antes era 0.2

    classificador = Sequential()           # estamos criando agora nossa rede neural

    '''
    ######################################################################################################
                                 CNN     
    ######################################################################################################
    '''

    '''
    if use_cnn:
        #tratamento da entrada para a CNN
        previsores_treinamento = np.expand_dims(previsores_treinamento, axis=2)
        previsores_teste = np.expand_dims(previsores_teste, axis=2)   
   
        classificador.add(Conv1D(50, 100, padding='same', activation='relu', input_shape=(numero_features, 1)))
        classificador.add(MaxPooling1D())
        classificador.add(Dropout(0.5))
        classificador.add(Conv1D(100, 100, padding='same', activation='relu'))
        classificador.add(MaxPooling1D())
    
        classificador.add(Flatten())
    '''

    '''
    ######################################################################################################
                                 Rede Neural     
    ######################################################################################################
    '''

    # teoricamente a RN começa no Sequential()
    classificador.add(Dense(units=100, activation='relu',
                            kernel_initializer='random_uniform', input_dim=numero_features))  # entrada. Normal: 27520
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units=160, activation='relu',
                            kernel_initializer='random_uniform'))  # camada oculta
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units=200, activation='relu',
                            kernel_initializer='random_uniform'))  # camada oculta
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units=160, activation='relu',
                            kernel_initializer='random_uniform'))  # camada oculta
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units=120, activation='relu',
                            kernel_initializer='random_uniform'))  # camada oculta
    classificador.add(Dropout(dropout))
    # camada de saida com 5 neuronios, um para cada classe
    # softmax retorna uma probabilidade para cada um dos rotulos
    classificador.add(Dense(units=numero_de_algoritmos, activation='softmax'))
    # vamos especificar a funcao que vamos usar para ajuste dos pesos, descida do gradiente e tal

    # com lr=0.00001 deu acuracia de 13.4%
    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.0, nesterov=False)
    adam = keras.optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
    # parece o melhor --> usava esse com lr = 0.0001
    adagrad = keras.optimizers.Adagrad(lr=0.01)
    adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95)
    nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)
    adamax = keras.optimizers.Adamax(
        lr=0.0005, beta_1=0.9, beta_2=0.999)  # default eh lr=0.002

    # adagrad com lr 0.0001 --> eh o que tava usando
    classificador.compile(optimizer=adamax, loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])  # para duas classe eh binary

    '''
    #antigo... o fit() precisa do dataset todo na memoria 
    history = classificador.fit(previsores_treinamento, classe_treinamento,
                            batch_size = 10, epochs = epoch_count,
                            validation_data = (previsores_teste,classe_teste),
                            shuffle = True, verbose = 1)  
    '''

    # to aqui
    # to aqui
    # to aqui

    # batch_size = 21  # *********

    tamanho_inputs_treinamento = len(filenames_treinamento)
    # numero total de amostras dividido pelo tamanho do batch
    steps_per_epoch_treinamento = int(tamanho_inputs_treinamento // batch_size)
    tamanho_inputs_teste = len(filenames_teste)
    steps_per_epoch_teste = int(tamanho_inputs_teste // batch_size)

    history = classificador.fit_generator(generator=gerador(filenames_treinamento, classe_treinamento, batch_size, numero_features),
                                          steps_per_epoch=steps_per_epoch_treinamento,
                                          epochs=epoch_count,
                                          verbose=1,  # mostra o progresso de treinamento em cada epoca
                                          #validation_data = gerador(filenames_teste,classe_teste,batch_size,numero_features),
                                          #validation_steps = steps_per_epoch_teste
                                          )
    ## OBS.: nao usei validation junto pra depois nao daria pra gerar matriz de confusao #
    # print(history.history.keys())

    '''
    ######################################################################################################
                                 se tudo der certo, apagar isso aqui    
    ######################################################################################################
    
    fit_generator(self, gerador_por_semente, samples_per_epoch, nb_epoch, verbose=1, callbacks=None, 
                  validation_data=None, nb_val_samples=None, class_weight=None, max_q_size=10, 
                  nb_worker=1, pickle_safe=False, initial_epoch=0)

    model.fit_generator(generator=my_training_batch_generator,
                   steps_per_epoch = int(3800 // batch_size),
                   epochs = 2,
                   verbose = 1,
                   validation_data = my_validation_batch_generator,
                   validation_steps = int(950 // batch_size))
    '''
    #  APAGAR ATE AQUI

    '''
    ######################################################################################################
                                 Plotando Curvas de Aprendizado      
    ######################################################################################################
    '''

    # visualizing losses and accuracy
    train_loss = history.history['loss']
    #val_loss   = history.history['val_loss']
    train_acc = history.history['categorical_accuracy']
    #val_acc    = history.history['val_categorical_accuracy']
    xc = range(epoch_count)

    # para printar no linux:
    print('train_loss: ', train_loss)
    #print('val_loss: ',val_loss)
    print('train_acc: ', train_acc)
    #print('val_acc: ',val_acc)

    '''
    #plt.figure()
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    #plt.title('Curva de Aprendizado - Erro')
    plt.ylabel('Loss') #loss
    plt.xlabel('Epoch') #epoch
    plt.legend(['Training', 'Test'], loc='upper left')
    plt.xlim(0,num_epocas)
    plt.show()
    #plt.figure()
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    #plt.title('Curva de Aprendizado - Acuracia')
    plt.ylabel('Accuracy') #accuracy
    plt.xlabel('Epoch') #epoch
    plt.legend(['Training', 'Test'], loc='upper left')
    plt.ylim(0.5,1.1)
    plt.xlim(0,num_epocas)
    plt.show()
    '''

    # testando gerar tudo junto:
    '''
    #plt.figure()
    plt.plot(xc, train_loss,color='red',label="Training Loss")
    plt.plot(xc, val_loss,color='orange',label="Test Loss")
    #plt.title('Curva de Aprendizado - Erro')
    plt.legend(['Training', 'Test'], loc='upper left')
    #plt.figure()
    plt.plot(xc, train_acc,color='green',label="Training Accuracy")
    plt.plot(xc, val_acc,color='blue',label="Test Accuracy")
    #plt.title('Curva de Aprendizado - Acuracia')
    plt.ylabel('Accuracy / Loss') #accuracy
    plt.xlabel('Epoch') #epoch
    plt.legend(loc='center right')  #upper right
    plt.ylim(0,1.05)
    plt.xlim(0,num_epocas)
    plt.savefig('CurvaAprendizado_8_SAs_200epocas_fold5.pdf', format='pdf')
    plt.show()
    '''

    '''
    #isso aqui faz a mesma coisa do predict()
    resultado = classificador.evaluate(previsores_teste, classe_teste)
    #print('Perda = ',resultado[0])
    #print('Acuracia = ',resultado[1])
    OBS.: para fit_generator() existe o evaluate_generator()... cheguei a usa-lo mas depois fui no esquema do predict() mesmo
    #exemplo de uso: 
    # scoreSeg = classificador.evaluate_generator(gerador(filenames_teste,classe_teste,batch_size,numero_features),steps_per_epoch_teste)
    '''

    '''
    ######################################################################################################
                                 se tudo der certo, apagar isso aqui    
    ######################################################################################################
    #teste novo
    #Confution Matrix and Classification Report
    #Y_pred = cla.predict_generator(validation_generator, num_of_test_samples // batch_size+1)
    y_pred = classificador.predict_generator(gerador(filenames_teste,classe_teste,batch_size,numero_features),steps_per_epoch_teste)

    # isso aqui ta certo !
    #A funcao abaixo retorna o indice da coluna que tem o maior valor. Alternativa: classe_teste2 = [np.argmax(t) for t in classe_teste]    
    classe_teste2 = np.argmax(classe_testaxis=1)

    #isso aqui ta errado ! 
    #previsoes     
    predict = classificador.predict_generator(gerador(filenames_teste,classe_teste,batch_size,numero_features),1)
    predict2 = np.argmax(predict, axis=1)

    type(classe_teste2)
    type(predict2)
    
    acuracia1 = accuracy_score(classe_teste2, predict2) 
    matriz1 = confusion_matrix(predict2, classe_teste2)   
    '''
    # apagar ate aqui

    # lembrando que para testes/validacao eu carrego tudo na memoria... old scheme
    previsores_teste = fase_de_testes_ou_validacao(
        filenames_teste, numero_features)

    # gero minhas previsoes
    previsoes = classificador.predict(previsores_teste)

    # a funcao abaixo retorna o indice que tem o maior valor.
    # pra entender melhor, bota classe_teste e classe_teste2 lado a lado
    classe_teste2 = [np.argmax(t) for t in classe_teste]
    # faco o mesmo para previsoes
    previsoes2 = [np.argmax(t) for t in previsoes]

    # vamos fazer uma gambiarra aqui agora
    if fold == 1:
        # aqui eh a precisao na base de dados de teste!
        acuracia1 = accuracy_score(classe_teste2, previsoes2)
        matriz1 = confusion_matrix(previsoes2, classe_teste2)
        print('Acuracia: ', acuracia1)
        print('Matriz de confusao: \n', matriz1)
    elif fold == 2:
        # aqui eh a precisao na base de dados de teste!
        acuracia2 = accuracy_score(classe_teste2, previsoes2)
        matriz2 = confusion_matrix(previsoes2, classe_teste2)
        print('Acuracia: ', acuracia2)
        print('Matriz de confusao: \n', matriz2)
    elif fold == 3:
        # aqui eh a precisao na base de dados de teste!
        acuracia3 = accuracy_score(classe_teste2, previsoes2)
        matriz3 = confusion_matrix(previsoes2, classe_teste2)
        print('Acuracia: ', acuracia3)
        print('Matriz de confusao: \n', matriz3)
    elif fold == 4:
        # aqui eh a precisao na base de dados de teste!
        acuracia4 = accuracy_score(classe_teste2, previsoes2)
        matriz4 = confusion_matrix(previsoes2, classe_teste2)
        print('Acuracia: ', acuracia4)
        print('Matriz de confusao: \n', matriz4)
    else:
        # aqui eh a precisao na base de dados de teste!
        acuracia5 = accuracy_score(classe_teste2, previsoes2)
        matriz5 = confusion_matrix(previsoes2, classe_teste2)
        print('Acuracia: ', acuracia5)
        print('Matriz de confusao: ')
        print(matriz5)

    porcentagem_features = 100
    print('Simulacao feita na topologia ', topology.name, ' com ', numero_de_algoritmos, 'algoritmos, ', seed_count,
          'sementes, ', porcentagem_features, '% de features ', epoch_count, 'epocas e batch size = ', batch_size)

    ######################################################################################################
    # Salvando em um arquivo externo
    ######################################################################################################

    # salvando o classificador em um arquivo .sav com pickle
    try:
        import cPickle as pickle
    except:
        import pickle
    #pickle.dump(classificador, open('classificador_RMLSA_usanet.sav', 'wb'))

    # salvando o classificador em um arquivo .sav com pickle
    if topology == Topology.USANET:
        classificador.save("classificador_RMLSA_usanet.h5")
    elif topology == Topology.PANEURO:
        classificador.save("classificador_RMLSA_paneuro.h5")

    ######################################################################################################
        # FIM
    ######################################################################################################

    fim = time.time()
    print('\nTempo da execucao: ', round((fim-inicio)/60), ' minutos')
