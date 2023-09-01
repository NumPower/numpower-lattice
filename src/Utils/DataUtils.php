<?php

namespace NumPower\Lattice\Utils;

use \NDArray as nd;

class DataUtils
{
    /**
     * @param \NDArray $x
     * @param float $epsilon
     * @return \NDArray
     */
    public static function quickNormalize(\NDArray $x, float $epsilon = 1e-8): \NDArray
    {
        $t = nd::transpose($x);

        $mean = nd::zeros([count($t), 1]);
        $std = nd::zeros([count($t), 1]);
        foreach ($t as $row => $sample) {
            $mean[$row] = nd::average($sample);
            $std[$row] = nd::std($sample);
        }

        $std = nd::clip($std, $epsilon, 1 - $epsilon);
        return nd::transpose(($t - $mean) / $std);
    }

    /**
     * Partition data in mini batches and shuffle it
     *
     * @param \NDArray $x
     * @param \NDArray $y
     * @param int $batchSize
     * @param float $validationSplit
     * @return array
     */
    public static function shuffleAndPartition(\NDArray $x, \NDArray $y, int $batchSize, float $validationSplit = 0.0): array
    {
        $rtn_payload = [
            "total_batches" => 0,
            "batch_size" => $batchSize,
            "batches_x" => [],
            "batches_y" => []
        ];
        if (count($x) < $batchSize) {
            $rtn_payload["total_batches"] = 1;
            $rtn_payload["batch_size"] = count($x);
            $rtn_payload["batches_x"][] = $x;
            $rtn_payload["batches_y"][] = $y;
            return $rtn_payload;
        }

        $rest = count($x) % $batchSize;
        $rtn_payload["total_batches"] = (int)(count($x) / $batchSize);
        if ($rest) {
            $rtn_payload["total_batches"] += 1;
        }

        for ($i = 0; $i < $rtn_payload["total_batches"]; $i++) {
            $rtn_payload["batches_x"][] = $x->slice([$i * $batchSize, ($i * $batchSize) + $batchSize]);
            $rtn_payload["batches_y"][] = $y->slice([$i * $batchSize, ($i * $batchSize) + $batchSize]);
        }

        print_r($rtn_payload);
        return $rtn_payload;
    }
}
