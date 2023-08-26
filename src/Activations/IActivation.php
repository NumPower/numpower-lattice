<?php

namespace NumPower\Lattice\Activations;

interface IActivation
{
    function activate(\NDArray $x): \NDArray;
    function derivative(\NDArray $x): \NDArray;
}