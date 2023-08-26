<?php

namespace NumPower\Lattice\Losses;

interface ILoss
{
    function calculate(\NDArray $target, \NDArray $output): \NDArray|float;
}