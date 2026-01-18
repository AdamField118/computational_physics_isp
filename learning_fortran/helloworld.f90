program helloworld
    implicit none
    character(len=13) :: hello_string
    
    hello_string = "Hello, world!"
    write(*,*) hello_string
end program helloworld