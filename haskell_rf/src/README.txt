Final Project Writeup

Michael LeMay

What I set out to make (skip to "What I made (specifically in Haskell)" if
this isn't relevant):

I created a random forest classifier in Haskell. You probably know what a random
forest classifier is, but just in case you don't, here's a (skippable) explanation.

A random forest is a machine learning classifier that consists of a set of
decision trees. Each decision tree is a tree (in my implementation binary)
with a threshold value based on a certain coordinate of an observation. So let's
say that in a dataset with height and weight as measurements, my tree has a threshold
of 130 on the weight of whatever observation. So observations that weight less
than 130 will go to the left, and observations that weight more will go to
the right. This repeats recursively until you reach a stopping point (more on
that later).

At the leaves, you then classify your leaf as predicting one of the classes.
This is done usually by finding the plurality class of observations at that
leaf in the training data, then making that leaf predict that class.

The decision tree is not particularly complicated, it is just a tree that
essentially asks a series of queries about values within an observation and
drops an observation to the right leaf, where the class is then predicted. A random
forest is just a set of decision trees, where the trees either vote or their value
is averaged to determine the predicted class of the overall random forest. Decision
trees on their own often have issues with overfitting to training data. So using
a random forest takes care of this issue by aggregating that overfitting. Each
decision tree is trained with data randomly sampled with replacement. This means
that each new tree is different, it is built on a different sample of data, and
this takes advantage of that random variance plus the overfitting to build
a significantly more accurate aggregated classifier.

All of this, to me, screamed Haskell. Aggregation of a data type into a larger data
type. A recursive tree based predictor. This, along with maybe an SVM, is one of the
only machine learning algorithms that feels pretty natural to Haskell. It is pretty
easy to see how it related to both recursion, higher order functions, and even referential
transparency, given unlike other machine learning methods you are not adjusting one
set of weights, but instead accumulating (folding together?) a larger structure
from a series of structures.

I've obviously skipped over how a decision tree is trained. Essentially, at each
level, the decision tree node needs to decide first if it's done or not. To decide
if you're done, usually a minimum point count or maximum depth are used. If done, it
calculates what node it's predicting at this leaf. If it is not done, then we have
to decide what the best possible split is at this node.

The best possible split is found by exhaustively checking every split (yes, sadly
it has to be done this way...). This involves checking every value of every dimension.
For each dimension, a theoretical split is made, and then the Gini index (a measure
of inequality between classes) is calculated. The smallest gini coefficient
is the one we want.

So that summarizes what I set out to make.

Some sources I used:

https://galton.uchicago.edu/~amit/Papers/shape_rec.pdf
https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

What I made (specifically in Haskell):

I implemented all that fine, with a working data entry reader that works
with csv files. Here's a quick rundown of the files I implemented, and
what typeclasses were used.

Main.hs - What you'd expect, a series of functions to take in a csv and check
          that csv for acceptability. If it's acceptable, creates the forest,
          trains it on the training data, predicts on the test data, and outputs
          the accuracy for both.

DataSet.hs - Simple implementation of a data set. A data set is a vector of any
             type, the data, and a vector of Int's, the classes. DataSet implements
             Functor, Applicative, Semigroup, and Monoid. It also contains a method to randomly
             shuffle given data, which I need for training of decision trees. Note that
             though DataSet is defined with generic data, I use Doubles as the data
             for simplicity, since Haskell does not easily allow for lists or vectors
             of different types, so restricting to data to doubles from Main makes
             sense.

DecisionTree.hs - Decision Tree is a class that implements the functionality described
                  above. It can predict, calculate a majority label for a vector (helper function
                  used in a few places), return an empty decision tree, and predict.
                  It does not make sense as a monoid because you cannot combine two
                  decision trees to get a new decision tree. You cannot apply functions
                  over the tree, so functor/applicative doesn't make sense. And you
                  never need to map over the entire tree, one is only ever interested
                  in dropping down the tree, rather than visiting every node.

RandomForest.hs - Random Forest is a class that implements the functionality described
                  above. It can add a new decision tree and predict. It is essentially
                  a grouping of decision trees. It implements Monoid and Semigroup. It
                  doesn't make sense to apply functions to RandomForest since it
                  doesn't take in unknown types. Monoid and semigroup make sense, however,
                  since an empty random forest makes sense as a concept, as does combining
                  two random forests. All you have to do is group the trees together.

I wrote why I chose the typeclasses I chose above. Essentially I implemented
every typeclass I think reasonably applied to any of the three datatypes
I defined. That resulted in foldable/applicative/monoid/semigroup for dataset
and monoid/semigroup for random forest.

I believe this, 6 typeclasses, should be sufficient for the assignment.

I also think the modularity for this assignment is straightforward.

Advanced features:

I decided to parallelize my Haskell code, since I'd always been really curious
about parallel functional code. Overall, I found the code pretty engaging to
write and intuitive, though I'll talk about results later on in my experimental
section.

Random forests are natural vessels for concurrency because operations inside the
decision tree creation can be parallelized, but more importantly, the building of
a decision tree is independent of any other decision trees. Therefore, it is easy
conceptually to build decision trees on many cores simultaneously, with no
penalty or issue with shared resource incurred for doing so. In other words, random
forests can be trained separately and independently, then, since they conform to
the typeclass monoid, subforests can be easily joined together.

Haskell divides it's parallelism into pure parallelism and concurrency. My program
tries to take advantage of both. Of course, in order to judge how effectively this
actually works, for both, I implemented both pure parallelism and concurrency. I
did this separately so that I could ascertain which was associated with performance
gains. So I have separate measurements for each and for both combined.

The pure parallelism uses parallel versions of otherwise linear executions. For instance,
this uses the par and pseq functions, along with parallelized mapping, all from
Control.Parallel. Here the functions determine whether the parallel overhead is
worth incurring, and run the operations on multiple cores if those cores are
available. The forest is still constructed linearly, however.

The concurrent version uses the linear implementation of DecisionTree and RandomForest,
but trains the random forests concurrently, again since each subforest is independent of
others and can be easily merged (due to it being a monoid).

Source for parallel Haskell:

A variety of wikis but (https://simonmar.github.io/bib/papers/strategies.pdf)
was very helpful.

My final version takes advantage of both pure parallelism and concurrency.

The parallelism consists of using Haskell parallel functions when appropriate
in DecisionTreeP.hs, RandomForestP.hs, and MainP.hs. This is pretty simple to
understand. Functions that use this parallelism are in files with a P appended
and in methods with "Parallel" appended to the end.

The concurrency consists of explicit threading in Main.hs. This was done for
both Main.hs and MainP.hs. If only one thread is supplied, only one thread will
be used. Otherwise, so long as the number of threads divides evenly into the number
of trees, the random forest will be divided and trained on the given number
of threads.

The creation of redundant _P.hs files is, I admit, a bit unclean. However, it was
easier to implement than passing along a "parallelizing" flag and invoking
different methods depending on the flag. So chalk it up to a senior trying to finish
this thing on time/know I am aware it is not a great design pattern.

How to run my code:

First, you need a data csv of entirely numerical values. I have provided two
such csv's, but if you want you can find your own, I'd recommend Kaggle. If you
want solidly accurate results, your csv needs to be > 1000 observations or so.

If your csv is too large, you will probably not be getting results anytime soon. I
found about 1000-2000 lines was about what my classifier could handle in a reasonable amount
of time (30 seconds to 10 minutes).

My sample csv's are in:
graduate-admissions/Admission_Predict.csv
heart-disease-uci/heart.csv
3 is the predictor column index for Admission_Predict, 30 for credit card fraud, 13 for heart disease

Command line parameters/command to run:

ghc -O2 -threaded -XBangPatterns --make Main[P].hs
./Main[P] [path to csv file] [percentage of data to use] [column of predictor (0 indexed)] [size of the forest] [number of threads]

Sample Run:

ghc -O2 -threaded -XBangPatterns --make MainP.hs
./MainP graduate-admissions/Admission_Predict.csv 1.0 3 20 4

The number of threads must evenly divide into the number of trees. I used either
20 or 30 trees for testing the model, and I'd recommend doing that as well.

All user interface in my program is through the command line.

Brief performance writeup:

Before I ran my tests, I was a little skeptical that my parallelized Haskell
code would have a meaningful improvement due to evaluation semantics. I wasn't
sure that my tree creation would be strictly evaluated on building, or how
that would work. I thought concurrency might offer some gains, but I was pretty
certain that parallelism would not offer any improvements, since there are very
few computationally intensive parallelizable tasks in decision tree training.

I performed my tests on a iMac Mini (2018). It's an 3.2GHz intel i7 with
six cores. So I stopped my tests at 6 cores.

I conducted the tests building 20 trees (or 24 for 6 threads) on the
Admission_Predict.csv dataset. I used 100% (1.0) of the data.

On purely linear code, I observed an average runtime of 24.1 seconds.

On purely parallelized code, with no concurrency, I observed an average runtime of
25.4 seconds. However, this had particularly high variance. Some trials were
faster than purely linear code (22-23 seconds) some trials were significantly slower
(25-28 seconds).

On purely concurrent code, with no parallelization, I observed runtimes of:
2 threads - 24.1 seconds
4 threads - 24.2 seconds
6 threads - 28.0 seconds

With both concurrency and parallelism, I observed runtimes of:
2 threads - 23.8 seconds
4 threads - 25.1 seconds
6 threads - 25.7 seconds (!!)

So to summarize, I saw slight performance gains with both concurrency and
parallelism, worse performance with just parallelism, and essentially the
same performance with just concurrency.

I am pretty unsurprised by these results. The only particularly noteworthy gain
was on 6 threads, 24 trees and it might have just been randomness. Essentially,
parallelizing individual operations and making the training concurrent did not
have the positive effect one might expect.

Why? Adding some print statements upon the completion of training for each
tree in the forest makes it pretty clear what is happening. The decision tree
training is being parallelized. However, due to both Haskell's lazy evaluation and
the recursive, interwoven structure of the decision tree and random forest, what is
actually happening is that the threads are not doing any real evaluation at all.
The threads are building up a computation, but they are not actually computing any of it.
The computing is then left to the main thread when it calculates the predicted values.

Upon prediction, the decision trees are then fully trained. That's why, even though
training should be much more time intensive than prediction, prediction is where
the program spends the most time.

So this issue is why I get no real performance gains from concurrent training of
the decision trees.

As for parallelism, the issue is that very few of the operations in decision tree
training are particularly computationally intensive. There are simply a lot of
low cost operations (exhaustive search). So it is essentially not worth it
to throw more cores at these simpler problems. The overhead imposed from trying
to parallelize exceeds the meager benefits of parallelizing.

This would of course be less of the case as the size of the data and the number of
trees in the forests increase. And you actually do see better performance for the
parallelized code with more trees and larger datasets (I experimented with this
as well, it was born out on 120 trees with 6 threads versus 1 thread).
I would speculate the most beneficial parallelization is the parallelized
mapping that I use in a few different places. That would have a far greater impact than
attempts to parallelize splitting in a decision tree, for instance.

So, if you want to see real strong performance gains, you would need to figure out
how to override Haskell's lazy evaluation system. I did some work on this, I'm using
bang patterns to try to force more evaluation and that resulted in the threads performing
significantly better, so perhaps I am just missing a few of those patterns.
I ran out of time for figuring out how to get it all to work (I might
work a bit more on this after the Friday deadline).

Forcing evaluation in the threads by printing out the trees did have an
effect on the concurrency performance. I saw gains of approximately 10%. Small, but
not nothing. I ran out of time to test this further, but essentially I can see that
evaluating in multiple threads did have a positive effect, when I used printing
to force evaluation within the worker threads. This effect can be seen in
Main.hs when line 120 is uncommented, so that the trees print out as they
are built.

Interestingly enough, using the parallelism and concurrency at the same time
reduced overall performance. Multiple threads were used to evaluate, but they seemed to be working
serially. Whether this was an effect of my computer, I tested this on my laptop that
has fewer cores, or that I was still missing something is unknown at the present time.
I speculate that using parallel methods consumes my computer's resources, making the
threads operate in serial. In other words, Haskell prioritizes parallelism over
concurrency, so given my machine only has 4 cores, the concurrency just ends up
executing serially, while doing just concurrency does see a performance gain with
throwing more threads at the problem. To see that the threads execute serially,
uncomment line 127 (and other comments in that function) to see that the trees
each take 5.5 seconds to be produced.

Essentially, I did figure out a hack to force evaluation inside the threads.
It has nice effects. If I have time, I'll update this section further to reflect
what more I learned about Haskell concurrency/parallelism.

So in general, I'd say that while my parallelized Haskell was not a performance success,
it was a good learning experience about both the benefits (the parallel code was quite
intuitive to write) and hazards (evaluation semantics make actually getting performance
gains a bit tricky, particularly for parallelism that isn't based around a simple return
value but on building a data structure) of parallelism in Haskell.

Libraries used:
Typeclass libraries (Control.Applicative... etc)
Control.Parallel

I know this writeup wasn't brief.

Thanks for reading all this and for a great quarter!
