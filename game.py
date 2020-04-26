import numpy as np
import json
from train_test import dl_model


#class responsible for actual gameplay
class Game:

    def __init__(self, char_to_id_pth):

        #load character to id mapping dumped by dataloader
        with open(char_to_id_pth, 'r') as f:
            self.char_to_id = json.load(f)

        #initialise model
        self.dl_model = dl_model('test_one')

    def play(self):

        #store correctly and incorresclty predicted characters
        misses, hits = [], []
        print("Instructions: For every predicted letter,")
        print("1. Type 'x' if letter does not exists in your word.")
        print("2. If it does exist, input the places at which it exists.")
        print("For e.g., if the word is hello and model's prediction is 'l', type 3 4 in the prompt. If prediction is 'a', type x.\n")

        print("Think of a word and input the number of characters:")
        num_chars = int(input())
        #initialise target string with blanks
        predicted_string = ['*']*num_chars
        
        while 1:

            #get sorted predictions according to probability
            best_chars = self.dl_model.predict(predicted_string, misses, self.char_to_id)
            #best char is the one with highest probability AND which does not belong to list of incorrectly predicted chars
            #AND not already present in the target string
            for pred in best_chars:
                if pred not in misses and pred not in predicted_string:
                    best_char = pred
                    break
            
            #predict and ask user for feedback
            print("Prediction: " + best_char + "\nWhat do ya think?")

            while 1:
                #get input
                inp = input().strip()
                if inp == 'x': #denotes character not present
                    output = 0
                    break
                try:
                    #if it is present, user returns a list of indices at which character is present (note that indexing begins from 1 for user feedback)
                    output = [int(x) for x in inp.split(' ')]
                    #update target string
                    for pred in output:
                        predicted_string[pred-1] = best_char
                    break
                except:
                    print("Invalid format. Please refer to instructions.") 
                    continue


            if output == 0: #indicates miss
                print("Miss")
                #append to missed characters list
                misses.append(best_char)
            else:
                #correctly predicted
                if '*' in predicted_string: #indicates game is not yet over since we still have unknown characters in target string
                    print("Guess correct! New target: " + ''.join(predicted_string))
                    hits.append(best_char)
                else: #indicates game is over. Report number of misses and return
                    print("Game over. Predicted word: " + ''.join(predicted_string))
                    print("Total misses: ", len(misses))
                    return misses



a = Game('pickle/char_to_id.json')
a.play()
