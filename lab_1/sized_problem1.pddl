
(define (problem size1)(:domain blocksworld)

(:objects
    A B C - block
    )

(:init
    (ontable A) 
    (on C A)(clear C) 
    (ontable B)(clear B)
    (handempty)
    (small-block C)
    (big-block A)(big-block B)
)

(:goal (and
    (on B A)(ontable A)
    (on C B)
    (clear C)
    (handempty)    
))
)
