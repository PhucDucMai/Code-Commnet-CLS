"
I am a priority queue that keeps my elements sorted by a sort block.

Responsibility:
I store elements and always answer the smallest one first.
I keep the order stable for elements that compare as equal.

Collaborators:
I use a SortedCollection to hold my elements and a Semaphore to protect concurrent access.

Public API and Key Messages:
- add: anElement    inserts an element at its sorted position
- removeFirst    answers and removes the smallest element
- sortBlock: aBlock    changes the ordering criterion

Example:
	queue := OrderedQueue sortBlock: [ :a :b | a priority < b priority ].
	queue add: job1; add: job2.
	queue removeFirst.

Internal Representation and Key Implementation Points.
Instance Variables
	elements:	a SortedCollection holding the queued elements
	mutex:	a Semaphore used as a lock around add: and removeFirst
Insertion costs O(log n) because SortedCollection uses a binary search.

See also Heap and OrderedCollection.
"
Class {
	#name : 'OrderedQueue',
	#superclass : 'Object',
	#instVars : [
		'elements',
		'mutex'
	],
	#category : 'Demo-Collections',
	#package : 'Demo-Collections'
}

{ #category : 'adding' }
OrderedQueue >> add: anElement [
	"Insert anElement at its sorted position"
	^ mutex critical: [ elements add: anElement ]
]

{ #category : 'removing' }
OrderedQueue >> removeFirst [
	^ mutex critical: [ elements removeFirst ]
]
