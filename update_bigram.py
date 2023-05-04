bigram_filepath = 'data/ImaginedHandwriting/raw/handwritingBCIData/BigramLM/webTextSentences_tokenized-2gram-50000.arpa'

with open(bigram_filepath, 'r') as file:
  data = file.readlines()

  data[2] = 'ngram 1=50003\n'
  data.insert(6, '-7.303118\t<unk>\t0\n')

  with open(bigram_filepath, 'w') as file:
      file.writelines(data)

print("Done!")