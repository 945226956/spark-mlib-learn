package com.zgc.spark.Utils;

import java.io.File;

/**
 * some utils for me
 *
 * @author guocheng.zhao
 * @date 2017/6/7 11:25
 * @tel 13524779402
 * @email guocheng.zhao@hand-china.com
 */
public class MlibUtls {
    /**
     * delete file
     * @param path
     */
    public static void deleteAllFilesOfDir(File path) {
        if (!path.exists())
            return;
        if (path.isFile()) {
            path.delete();
            return;
        }
        File[] files = path.listFiles();
        for (int i = 0; i < files.length; i++) {
            deleteAllFilesOfDir(files[i]);
        }
        path.delete();
    }
}
