<?php

namespace NumPower\Lattice\Initializers;

interface IInitializer
{
    function initialize(): \NDArray;
}