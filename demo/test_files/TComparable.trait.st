"
I am a trait that gives comparison messages to any class defining <.

My users only implement < and they get <=, >, >=, max: and min: for free.

Example:
	Money uses: TComparable.
	(Money amount: 5) max: (Money amount: 7).

I am used by Money and Version.
"
Trait {
	#name : #TComparable,
	#category : #'Demo-Kernel'
}

{ #category : #comparing }
TComparable >> <= aComparable [
	^ (aComparable < self) not
]

{ #category : #comparing }
TComparable >> max: aComparable [
	^ self < aComparable ifTrue: [ aComparable ] ifFalse: [ self ]
]
