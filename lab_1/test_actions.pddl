(define (problem test_actions)(:domain blocksworld)

(:objects
    A B C - block
    )

(:init
    (holding A)
    (ontable B)(clear B)
    (ontable C)(clear C)
)

(:goal (and
    (ontable A)(clear A)
    (ontable C)   
    (on B C)(clear B)
    (handempty)
))
)