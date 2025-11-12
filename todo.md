TODO add timeout 10 min
TODO show no plots just export them
TODO make Exp2 fit based on examples and use a bandwidth for pink noise of 1Hz-100Hz and a timeseries of 5 samples with a sampling frequency of 300Hz! make sure the pinknoise is bandpassed to 1-50Hz and make sure the mass spring dampers peak is tuned to 25Hz just as in the julia examples! make the neural network 6 parameters in a 2x3 matrix just like in the julia example using dense layer without bias!
    - structure:
      - it should have all its functions in neural_ode_funcs.py
      - it should have all its tests in neural_ode_tests.py
      - it should run as a demo using neural_ode_example.py
      - all should be as simple as possible
    - tests
      - there shhould be unit tests for all funcs, integration tests and regression tests all based on speed and simplicity
      - the neural odes should fit as in the example
      - the neural odes should behave as described in on_neural_odes.pdf
    - goals 
      - make it as elegant as possible
      - make all tests work
      - make all todos work
    - docs
      - if puzzled on something use context7 with diffrax for docs
      - if puzzled on something go through doctorial thesis on_neural_odes.pdf for insights into neural odes and their working ways
TODO make sure the error of the fit of neural_ode_example in exp 2 is less than 10^-3

TODO make Exp3 which is exp2 just fitting on 50 samples!
    - structure:
      - it should have all its functions in neural_ode_funcs.py
      - it should have all its tests in neural_ode_tests.py
      - it should run as a demo using neural_ode_example.py
      - all should be as simple as possible
    - tests
      - there shhould be unit tests for all funcs, integration tests and regression tests all based on speed and simplicity
      - the neural odes should fit as in the example
      - the neural odes should behave as described in on_neural_odes.pdf
    - goals 
      - make it as elegant as possible
      - make all tests work
      - make all todos work
    - docs
      - if puzzled on something use context7 with diffrax for docs
      - if puzzled on something go through doctorial thesis on_neural_odes.pdf for insights into neural odes and their working ways

TODO make sure the error of the fit of neural_ode_example in exp 3 is less than 10^-3

TODO make Exp4 which is exp3 just making the parameters initialized using the real params with different levels of pertubations of noise!
    - structure:
      - it should have all its functions in neural_ode_funcs.py
      - it should have all its tests in neural_ode_tests.py
      - it should run as a demo using neural_ode_example.py
      - all should be as simple as possible
    - tests
      - there shhould be unit tests for all funcs, integration tests and regression tests all based on speed and simplicity
      - the neural odes should fit as in the example
      - the neural odes should behave as described in on_neural_odes.pdf
    - goals 
      - make it as elegant as possible
      - make all tests work
      - make all todos work
    - docs
      - if puzzled on something use context7 with diffrax for docs
      - if puzzled on something go through doctorial thesis on_neural_odes.pdf for insights into neural odes and their working ways

TODO make sure the error of the fit of neural_ode_example in exp 4 is less than 10^-3