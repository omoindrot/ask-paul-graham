# Ask Paul Graham

Retrieve the best paragraphs from all of Paul Graham's essays given a query.


## TODO

Dataset extraction:
- [ ] Specific rule for essay "How YCombinator started" (currently one big paragraph)
- [ ] Handle citations, which get aggregated into one paragraph ([example](http://www.paulgraham.com/really.html?viewfullsite=1#:~:text=A%20lot%20of%20founders%20complained))
- [ ] Create a real clean dataset with csv file?
- [ ] Parse the article date and remove it from paragraphs
- [ ] Add all tweets from Paul Graham? (32k tweets)

Query:
- [ ] Split out encoding script
- [ ] Add argparse or equivalent to handle parameters
- [ ] Add streamlit demo
- [ ] Display context around paragraph?
- [ ] Or add link to highlighted text in content ([example](http://www.paulgraham.com/really.html?viewfullsite=1#:~:text=A%20lot%20of%20founders%20complained))

Huggingface:
- [ ] Add cleaned dataset to HF?
- [ ] Add demo to HF Spaces?
- [ ] Tweet it to Paul Graham?
