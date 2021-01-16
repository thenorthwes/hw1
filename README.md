# Homework 1 NLP 517 HW 1

For simplicity this program runs all parts of the homework by executing `main.py`
- Running > `python` Python 3.8.2 (default, Nov  4 2020, 21:23:28) [Clang 12.0.0 (clang-1200.0.32.28)] on darwin
- `python main.py` 

Note this implementation DOES write back to the local disk 
- Provided data sets are contained within `hw1/files/****`
- Writing back takes place in `./workspace/****`

Time to complete averages 2 minutes to run the entire homework set
- Output is pretty printed as well as possible and should be consistent with the written report.
- Time to complete based on 2.7 GHz Intel Core i7-6820HQ
- This was developed in `PyCharm` and should be openable as a project a well

# Changing configuration
k/λ values can be changed in functions within `main.py`
- For example `part_four_two()` contains various configurations of λ
- k is configured in `part_two` as well as ngram size (although not requested)
- UNK thresholds are configured in `config.py` in `UNK_THRESHOLD` and `MAX_UNKS` 

# Homework components
Part 1 Leverages `unigrammodel.py` and `ngram.py`

Part 2 Leverages `ngrammodel.py`
- In part 2, k flags are used in `ngrammodel.py`

Part 4 leverages `linear_interpolation.py`

All perplexity formulas are done in `PerplexityScorer.py`
- Rounding is to 4 places for printing reasons

Some Unit tests are written -- mostly for working through some ideas as my understanding grew