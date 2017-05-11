# -*- coding: utf-8 -*-
"""
Created on Thu May 11 2017

@author: spranger
"""

import numpy as np
import tensorflow as tf
import logictensornetworks as ltn

from pylab import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ltn.default_layers = 16
ltn.default_smooth_factor = 1e-10
ltn.default_tnorm = "product"
ltn.default_aggregator = "min"
ltn.default_optimizer = "rmsprop"
ltn.default_clauses_aggregator = "min"
ltn.default_positive_fact_penality = 0

data_embeddings = { "horse" : [1.,0.],
                   "cow" : [-1.,1.],
                   "donkey" : [1.,0.],
                   "pony" : [.8,0.],
                   "bw" : [0.,1.],
                   "lion" : [-1.,-1.],
                   "zebra" : None}
predicates = {}
constants = {}
clauses = {}

embedding_dimension = 2

word = ltn.Domain( embedding_dimension, label="Word")


for w,embedding in data_embeddings.iteritems():
    p = ltn.Predicate( w, word)
    predicates[w] = p
    c = ltn.Constant( w, domain = word, value = embedding)
    constants[w] = c
    clause_label = "%s_%s" % (w,w)
    clauses[clause_label] = ltn.Clause( [ltn.Literal(True,p,c)],label=clause_label)

#all_words = ltn.Domain_union( constants.values())

clauses["notZorH"] = ltn.Clause([ltn.Literal(False, predicates["zebra"], word),
                                    ltn.Literal(True, predicates["horse"], word)],
                                    label = "notZorH")
clauses["notZorBW"] = ltn.Clause([ltn.Literal(False, predicates["zebra"], word),
                                    ltn.Literal(True, predicates["bw"], word)],
                                    label = "notZorBW")
clauses["notHornotBWorZ"] = ltn.Clause([ltn.Literal(True, predicates["zebra"], word),
                                    ltn.Literal(False, predicates["bw"], word),
                                    ltn.Literal(False, predicates["horse"], word)],
                                    label = "notHornotBWorZ")
clauses["restrictDomain"] = ltn.Clause( [ltn.Literal( True, ltn.In_range(word, [-1., -1.],[1., 1.]), constants["zebra"])],
                                        label = "restrictDomain")

clauses["notHlion"] = ltn.Clause( [ltn.Literal(False, predicates["horse"], constants["lion"])],
                                        label = "notHlion")
clauses["notBWlion"] = ltn.Clause( [ltn.Literal(False, predicates["bw"], constants["lion"])],
                                        label = "notBWlion")

data = np.array([[i,j] for i in np.linspace(-1,1,21,endpoint=True)
                for j in np.linspace(-1,1,21,endpoint=True)],
                dtype=np.float32)

feed_dict = { word.tensor:data }

KB = ltn.KnowledgeBase("Embedding", clauses.values(), "")

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sat_level = sess.run(KB.tensor, feed_dict = feed_dict)
print "initialization", sat_level

while sat_level < 1e-10:
    sess.run(init)
    sat_level = sess.run(KB.tensor, feed_dict = feed_dict)
    print "initialization",sat_level
print(0, " ------> ", sat_level)

for i in range(1000):
  KB.train(sess, feed_dict = feed_dict)
  sat_level = sess.run(KB.tensor, feed_dict = feed_dict)
  print(i + 1, " ------> ", sat_level)
  if sat_level > .99:
      break
#KB.save(sess)

print( "zebra: %s" % sess.run(constants["zebra"].tensor))
for c, cc in clauses.iteritems():
    print( "%s: %s" %( c, sess.run( cc.tensor, feed_dict = feed_dict)))

result = sess.run( ltn.Literal( True, predicates["zebra"], word).tensor, feed_dict = feed_dict)
plt.imshow( result.reshape( (21,21)).T)

result = sess.run( ltn.Literal( False, predicates["horse"], word).tensor, feed_dict = feed_dict)
plt.imshow( result.reshape( (21,21)).T)

print( "is the zebra a lion? %s" \
    % sess.run( ltn.Clause( [ltn.Literal( True, predicates["lion"], constants["zebra"])], label="bla").tensor))

print( "is the zebra bw? %s" \
    % sess.run( ltn.Clause( [ltn.Literal( True, predicates["bw"], constants["zebra"])], label="bla").tensor))

print( "is the zebra a horse? %s" \
    % sess.run( ltn.Clause( [ltn.Literal( True, predicates["horse"], constants["zebra"])], label="bla").tensor))


print( sess.run( ltn.Clause( [ltn.Literal( False, predicates["zebra"], word),
                       ltn.Literal( False, predicates["lion"], word)]).tensor, 
                       feed_dict = feed_dict))
#sess.close()