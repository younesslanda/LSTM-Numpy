#Author : Youness Landa
import numpy as np

from sgd import SGD
from model import TwoLayersLSTM

def main():
    data = open('alllines.txt', 'r').read() #Shakespeare corpus
    chars = list(set(data))

    data_size, vocab_size = len(data), len(chars)

    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    #convert all the data to a list of indexes
    data = list(data)
    for i, ch in enumerate(data):
        data[i] = char_to_ix[ch]

    #Hyperparams	
    epochs = 100 
    seq_len = 100
    test_seq_len = 50
    hidden_dim = 100
    lr = 0.005

    sgd = SGD(lr=lr)
    model = TwoLayersLSTM(input_dim=vocab_size, hidden_dim=hidden_dim)

    loss_epochs = []
    for i_epoch in range(1, epochs+1):
        data_ptr = np.random.randint(100)

        n = 0
        running_loss = 0

        while True:
            input_seq = data[data_ptr : data_ptr + seq_len]
            target_seq = data[data_ptr + 1 : data_ptr + seq_len + 1]
            
            # forward pass
            x = np.zeros((len(input_seq), vocab_size))
            for t in range(seq_len):
                x[t][input_seq[t]] = 1
                
            out, fwd_x, fwd_state1, fwd_state2 = model.forward(x)
            
            # compute loss
            loss = model.loss(out, target_seq)
            running_loss += loss
            
            # compute gradients and optimize
            y_target = np.zeros((len(target_seq), vocab_size))
            for t in range(seq_len):
                y_target[t][target_seq[t]] = 1

            # the deltas of the cross entropy loss is y_hat(t) - y_target(t)
            deltas = out - y_target

            dx1, gradients, gradients1, gradients2 = model.backward(deltas, fwd_x, fwd_state1, fwd_state2)
            gradients, gradients1, gradients2 = model.clip(gradients, gradients1, gradients2, clip_value=1)

            model.optimize(sgd, gradients, gradients1, gradients2)
            
            # update the data pointer
            data_ptr += seq_len
            n +=1
            
            # if at end of data : break
            if data_ptr + seq_len + 1 > data_size:
                break

        loss_epochs.append(running_loss/n)
        print("Train epoch: {0} \t Loss: {1:.8f}".format(i_epoch, running_loss/n))

        # sample / generate a text sequence after every epoch
        # random character from data to begin
        data_ptr = 0

        rand_index = np.random.randint(data_size-1)
        input_seq = data[rand_index : rand_index+1]
        
        print('Test phase')
        print("-" * 10)
        while True:
            # forward pass
            x = np.zeros((1, vocab_size))
            x[0][input_seq[0]] = 1
                
            out, fwd_x, fwd_state1, fwd_state2 = model.forward(x)

            # sample next index
            ix = np.random.choice(range(vocab_size), p=out.ravel())
            #next input x
            x = np.zeros((1, vocab_size))
            x[0][ix] = 1
            
            # print the sampled character
            print(ix_to_char[ix], end='')
            
            # next input is current output
            data_ptr += 1
            
            if data_ptr > test_seq_len:
                break
            
        print("-" * 10)