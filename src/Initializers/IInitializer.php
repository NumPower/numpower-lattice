<?php

namespace NumPower\Lattice\Initializers;

interface IInitializer
{
    function initialize(bool $use_gpu): \NDArray;
}