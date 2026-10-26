"
I am a fixed-size circular buffer. When I am full, adding a new element overwrites the oldest one.

Responsibility: I keep the most recent elements in insertion order and answer them without copying.

Collaborators: I use an Array to store my elements and OrderedCollection when converting with #asOrderedCollection.

Public API and Key Messages
- add: anObject    appends an element, dropping the oldest when full
- do: aBlock        iterates from the oldest to the newest element
- RingBuffer new: 10    creates a buffer that holds ten elements

Example: (RingBuffer new: 3) add: 1; add: 2; add: 3; add: 4; asArray ""returns #(2 3 4)""

Internal Representation and Key Implementation Points.
Instance Variables
	items:	the Array holding the elements
	head:	index of the oldest element; it moves forward modulo the capacity
"
Class {
	#name : 'RingBuffer',
	#superclass : 'Object',
	#instVars : [
		'items',
		'head',
		'size'
	],
	#category : 'Demo-Collections',
	#package : 'Demo-Collections'
}

{ #category : 'instance creation' }
RingBuffer class >> new: anInteger [
	^ self new setCapacity: anInteger
]

{ #category : 'adding' }
RingBuffer >> add: anObject [
	"Add anObject, overwriting the oldest element when full"
	items at: (head + size - 1 \\ items size) + 1 put: anObject.
	size < items size
		ifTrue: [ size := size + 1 ]
		ifFalse: [ head := head \\ items size + 1 ].
	^ anObject
]
