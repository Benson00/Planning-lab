
(define (problem size2)(:domain blocksworld)

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
    (ontable A)(clear A)
    (on B C)(clear B)
    (ontable C)
    (handempty)    
))
)
