<?php

namespace NumPower\Lattice\Layers;

use NumPower\Lattice\Activations\IActivation;

interface ILayer
{
    function initialize(ILayer $previousLayer, bool $use_gpu, bool $isOutput): void;
    function generateOutputShape(): array;
    function inputShape(): array;
    function forward(\NDArray $inputs): \NDArray;
    function backward(ILayer $previousLayer, \NDArray $error): \NDArray;
    function getActivation();
    function setActivationFunction(IActivation $function): void;
    function getActivationFunction(): IActivation;

    function setDerivative(\NDArray $derivative): void;

    function getDerivative(): \NDArray;

    function getWeights(): \NDArray;

    function setWeights(\NDArray $weights): void;

    function isOutput(): bool;
}