<?php

namespace NumPower\Lattice\Core\Layers;

use NumPower\Lattice\Core\Tensor;

interface ILayer
{
    /**
     * @param Tensor $inputs
     * @param bool $training
     * @return Tensor
     */
    public function __invoke(Tensor $inputs, bool $training = false): Tensor;

    /**
     * @param int[] $inputShape
     * @return void
     */
    public function build(array $inputShape): void;

    /**
     * @return int[]
     */
    public function generateOutputShape(): array;

    /**
     * @return int[]
     */
    public function getInputShape(): array;

    /**
     * @return string
     */
    public function getName(): string;

    /**
     * @return bool
     */
    public function built(): bool;

    /**
     * @return int
     */
    public function countParams(): int;

    /**
     * @return bool
     */
    public function isTrainable(): bool;

    /**
     * @return int
     */
    public function countTrainableParams(): int;

    /**
     * @return int
     */
    public function countNonTrainableParams(): int;

    /**
     * @return Tensor[]
     */
    public function getTrainableWeights(): array;

    /**
     * @param int $batchSize
     * @return void
     */
    public function setBatchSize(int $batchSize);

    /**
     * @return int
     */
    public function getBatchSize(): int;
}
