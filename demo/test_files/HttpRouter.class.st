"
I am a router that maps HTTP requests to handler blocks.

Responsibility:
I match the method and path of a request against my routes and call the first handler that fits.
I answer a 404 response when no route matches.

Collaborators:
I receive HttpRequest objects from the HttpServer and answer HttpResponse objects.

Public API and Key Messages:
- on: aMethod path: aPattern do: aBlock    adds a route
- handle: aRequest    answers the response of the matching route

Example:
	router := HttpRouter new.
	router on: #GET path: '/users/:id' do: [ :req | UserController show: req ].
	router handle: (HttpRequest get: '/users/42').

Internal Representation and Key Implementation Points.
Instance Variables
	routes:	an OrderedCollection of HttpRoute, checked in insertion order
Path patterns such as ""/users/:id"" are compiled once into a regular expression.
"
Class {
	#name : 'HttpRouter',
	#superclass : 'Object',
	#instVars : [
		'routes'
	],
	#category : 'Demo-Web',
	#package : 'Demo-Web'
}

{ #category : 'routing' }
HttpRouter >> on: aMethod path: aPattern do: aBlock [
	routes add: (HttpRoute method: aMethod pattern: aPattern handler: aBlock)
]
