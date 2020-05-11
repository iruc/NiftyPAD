
import os
import numpy as np
import nibabel as nib
import pandas as pd
import argparse
from scipy import stats
import matplotlib.pyplot as plt
from niftypad.cl_conversion.get_gaain_data import get_ad_yc_subjects, get_supplementary_tables, get_mni_pet
import niftypad


class TracerConversionToPib(object):
    def __init__(self, tracer='A', list_subjects_pib=[], list_ids_pib=[], list_subjects_tracer=[], list_ids_tracer=[]):

        self.check_conditions()
        # calculate pib_suvr_ind** and tracer_suvr_ind using standard CTX and WC VOIs
        self.calculate_suvr_ind_pib(tracer, list_subjects_pib, list_ids_pib)
        self.calculate_suvr_ind_tracer(tracer, list_subjects_tracer, list_ids_tracer)
        #
        #todo: poner si elegir standardr (slope_std y intercept_std) vois o no (slope_ns y intercept_ns)
        self.regression_suvr_ind_std_to_pib_calc()
        # todo: Add plotting of CL
        #self.convert_to_cl_tracer(tracer)

    def check_conditions(self):
        group_subjects = self.df_subjects_tracer['group'].values
        ad = np.where(group_subjects == 'AD')[0]
        nc = np.where(group_subjects == 'YC')[0]
        if len(group_subjects) >= 25 and len(nc) >= 10 and len(ad) >= 5:
            pass
        else:
            raise InterruptedError('Conditions of number of subjects needed for calibration not fulfilled')

    def calculate_suvr_ind_tracer(self, tracer, list_subjects, list_ids):
        #todo: if we only perform standard method (with standard CTX) no need of making different function for tracer
        cergy, whcer, cerbst, pons = self.calculate_suvr_ind(list_subjects, list_ids)
        #if tracer == 'A':
        #    self.calculate_suvr_ind_A(list_subjects, list_ids)
        self.df_subjects_tracer['GreyCerebellum'] = cergy
        self.df_subjects_tracer['WholeCerebellum'] = whcer
        self.df_subjects_tracer['WholeCerebellumBrainStem'] = cerbst
        self.df_subjects_tracer['Pons'] = pons

    def calculate_suvr_ind_pib(self, tracer, list_subjects, list_ids):

        cergy, whcer, cerbst, pons = self.calculate_suvr_ind(list_subjects, list_ids)
        #if tracer == 'A':
        #    self.calculate_suvr_ind_A(list_subjects, list_ids)
        self.df_subjects_pib['GreyCerebellum'] = cergy
        self.df_subjects_pib['WholeCerebellum'] = whcer
        self.df_subjects_pib['WholeCerebellumBrainStem'] = cerbst
        self.df_subjects_pib['Pons'] = pons

    def regression_suvr_ind_std_to_pib_calc(self, region):
        # fter plotting the PiBSUVrIND** values on the x-axis and the TracerSUVrIND values on the y-axis a slope
        # (TracermStd) and intercept (TracerbStd) is calculated, where the “Std” subscript desig- nates that the
        # standard CTX and WC VOIs were used

        if not np.all(self.df_subjects_pib['subjects'].values == self.df_subjects_tracer['subjects'].values):
            # make sure that the order of the subjects is the same in the
            to_reorder = self.df_subjects_pib['subjects'].values.argsort()[
                self.df_subjects_tracer['subjects'].values.argsort()]  # array([1
            # order_subjects = order_subjects[to_reorder]
            # calculated_cl = calculated_cl[to_reorder]
            self.df_subjects_tracer.reindex(to_reorder)

        self.suvr_pib_calc = self.df_subjects_tracer
        self.surrogate_slope_intercept_tracer = pd.DataFrame(columns= ['slope', 'intercept', 'r2'], index=self.regions_names)
        # todo: save values of Calibration of another tracer for future use as a surrogate reference

        for region in self.regions_names:
            suvr_pib = self.df_subjects_pib[region].values
            suvr_tracer = self.df_subjects_tracer[region].values
            slope_tracer_std, intercept_tracer_std, r_value = self.linear_correlation(suvr_pib.astype(float), suvr_tracer.astype(float))
            r2 = r_value**2
            if r2 < 0.7:
                raise Exception('### Level-2 calibrating not valid: no trasnformation to CL  ###')
            else:
                print('### Level-2 calibrating valid: saving slope and intercept values. Starting conversion to CL  ###')
            self.surrogate_slope_intercept_tracer.loc[region] = [slope_tracer_std, intercept_tracer_std, r2]

            # A TracermStd of 1.0 means the surrogate has the same spe- cific signal (or dynamic range) as PiB.
            # A slope of 0.5, half the signal of PiB, a slope of 2, twice the signal of PiB. Thus, the numerical value of
            # this slope is informative regarding the relative signals of PiB and the surrogate tracer. The conversion to
            # Centiloid units would then be accom- plished by first converting the TracerSUVrIND values into
            # “PiB calculated” SUVr values (PiB2CalcSUVrIND)
            pib_calc_suvr_ind = self.get_pib_calculated_suvr(suvr_tracer, slope_tracer_std, intercept_tracer_std)
            self.suvr_pib_calc[region] = pib_calc_suvr_ind

    def get_pib_calculated_suvr(self, suvr_tracer, slope_tracer_std, intercept_tracer_std):
        pib_calc_suvr_ind = (suvr_tracer - intercept_tracer_std) / slope_tracer_std
        return pib_calc_suvr_ind


    def convert_to_cl(self, tracer):
        # todo poner la correspondiente formula para cada tracer
        region = 'WholeCerebellum'
        pib_mean_ad, pib_mean_yc = self.calculate_mean_suvr(region)
        cl = []
        for pib_suvr_ind in self.df_subjects_gaain[region]:
            cl.append(self.formula_to_cl(pib_suvr_ind, pib_suvr_ad=pib_mean_ad, pib_suvr_yc=pib_mean_yc))
        self.df_subjects_tracer['pib_cl'] = cl

        # Perform QC for the CL conversion
        self.qc_linear_correlation_standard(cl, self.df_subjects_gaain['subjects'].values, region)


class CentiloidConversionPib(object):

    def __init__(self, list_subjects=[], list_ids=[]):
        #todo: input not specified, one list of paths
        #todo: poner en listas en vez de propias variables para rois?
        #todo: Input of subjects es folder name? o arrays?
        self.ref_table = get_supplementary_tables()

        self.calculate_suvr_ind(list_subjects, list_ids)
        self.qc_individual_suvr()
        self.convert_to_cl_gaain()

    def calculate_suvr_ind(self, list_subjects, list_ids):
        #todo: change according to input of subjects

        mask_cort, mask_ref_roi = self.get_mask_regions()
        cergy, whcer, cerbst, pons = [], [], [], []

        for pet, id in zip(list_subjects, list_ids):
            # assuming that subject is already a anumoy array
            val_cort = np.mean(pet[mask_cort])
            cergy.append(val_cort / np.mean(pet[mask_ref_roi[0]]))
            whcer.append(val_cort / np.mean(pet[mask_ref_roi[1]]))
            cerbst.append(val_cort / np.mean(pet[mask_ref_roi[2]]))
            pons.append(val_cort / np.mean(pet[mask_ref_roi[3]]))

        #suvr_ind = pd.DataFrame(data={'subjects': list_ids, 'GreyCerebellum': cergy, 'WholeCerebellum': whcer,
        #                              'WholeCerebellumBrainStem': cerbst, 'Pons': pons})
        self.df_subjects_gaain['GreyCerebellum'] = cergy
        self.df_subjects_gaain['WholeCerebellum'] = whcer
        self.df_subjects_gaain['WholeCerebellumBrainStem'] = cerbst
        self.df_subjects_gaain['Pons'] = pons

    def qc_individual_suvr(self, region='WholeCerebellum'):
        # step 13
        # get Supplementary Table 1,
        # calculate the percent difference from the CTX SUVr values published in the Centiloid paper
        '''
                :param table: From Unnmaed: 1 -  SUVr
                               From Unnamed: 5 - Scaled UNits
                '''

        #table_subjects = [value.strip() for value in self.ref_table['Supplementary Table 1.'].values[3:]]
        table_subjects = self.ref_table['Supplementary Table 1.'].values
        suvr_data = self.ref_table.iloc[:,1:5]
        #pos_region = np.where(region == )
        pos_region = 1
        for subject in self.df_subjects_gaain['subjects'].values:
            loc_subject = np.array(table_subjects) == subject
            suvr_data[loc_subject].values#[1,pos_region]
        #todo: finish this function: qc of individual subjects SUVr

    def convert_to_cl_gaain(self):
        # todo: calculate for all the regions
        region = 'WholeCerebellum'
        pib_mean_ad, pib_mean_yc = self.calculate_mean_suvr(region)
        cl = []
        for pib_suvr_ind in self.df_subjects_gaain[region]:
            cl.append(self.formula_to_cl(pib_suvr_ind, pib_suvr_ad=pib_mean_ad, pib_suvr_yc=pib_mean_yc))
        self.df_subjects_gaain['pib_cl'] = cl

        # Perform QC for the CL conversion
        self.qc_linear_correlation_standard(cl, self.df_subjects_gaain['subjects'].values, region)

    def formula_to_cl(self, pib_suvr_ind, pib_suvr_ad=2.076, pib_suvr_yc=1.009):
        '''
        :param pib_suvr_ind: individual’s SUVr value
        :param pib_suvr_ad: mean SUVr of the 45 AD-100 subjects.
        :param pib_suvr_yc: mean SUVr of the 34 YC-0 subjects
        :return: CL: converted centiloid value
        '''
        CL = 100 * (pib_suvr_ind - pib_suvr_yc) / (pib_suvr_ad - pib_suvr_yc)
        return CL

    def qc_linear_correlation_standard(self, calculated_cl, order_subjects, region):
        reference_cl_vois = self.ref_table.iloc[:,5:]
        chosen_region = 2 # region
        if not np.all(self.ref_table['Supplementary Table 1.'].values == order_subjects):
            # make sure that the order of the subjects is the same in the
            to_reorder = order_subjects.argsort()[self.ref_table['Supplementary Table 1.'].values.argsort()]  # array([1
            order_subjects = order_subjects[to_reorder]
            calculated_cl = calculated_cl[to_reorder]

        reference_cl = reference_cl_vois.iloc[:, chosen_region].values
        reference_cl.astype(float)
        slope, intercept, r_value  = self.linear_correlation(reference_cl.astype(float),calculated_cl.astype(float))

        # the slope will be between 0.98 and 1.02, the intercept will be between 22 and 2 CL and the R2 will be .0.98.
        r2 = r_value**2
        if r2 >= 0.98 and intercept >= -2 and intercept <= 2:
            self.firt_step = True
            print(' ### Standard method from calibrating site pipeline is validated ###')
        else:
            self.firt_step = False
            raise Exception ('### Standard method from calibrating site pipeline is not validated,'
                             ' redo normalizations steps ###')


class FullCalibrationProcedure(CentiloidConversionPib, TracerConversionToPib):

    def __init__(self, args):
        # crear FullConversionProcedure class que contenga join functions

        self.regions_names = ['GreyCerebellum', 'WholeCerebellum', 'WholeCerebellumBrainStem', 'Pons']
        # Reproduction of 1-Level analysis
        # todo: dowload beforehand data or download at the moment

        path_subjects_gaain = ['YC-0_MR/nifti', 'YC-0_MR/nifti'] # Path to already normalized subjects
        self.df_subjects_gaain = get_ad_yc_subjects(list_path=None, subjects_path=path_subjects_gaain)
        pet_subjects_gain, list_ids_gaain = get_mni_pet(path_subjects_gaain)
        self.get_voi_gaain('Centiloid_Std_VOI/nifti/2mm')
        CentiloidConversionPib.__init__(self, tracer=args.tracer_orig, path=None, list_subjects=pet_subjects_gain,
                                        list_ids=list_ids_gaain)

        if args.method == 'non-standard' and self.firt_step:
            path = 'pet folder path'
            pet_subjects, list_ids = get_mni_pet(path)
            self.df_subjects_tracer = get_ad_yc_subjects(list_path=None, subjects_path=path)
            TracerConversionToPib.__init__(self, tracer=args.tracer, list_subjects=pet_subjects,
                                        list_ids=list_ids)
            folder_to_save = os.path.join(os.path.dirname(niftypad.cl_conversion.__file__), args.tracer)
            self.surrogate_slope_intercept_tracer.to_csv(os.path.join(folder_to_save, 'calibration_'+ args.tracer +'.csv'))

    def calculate_suvr_ind(self, list_subjects, list_ids):
        #todo: change according to input of subjects

        mask_cort, mask_ref_roi = self.get_mask_regions()
        cergy, whcer, cerbst, pons = [], [], [], []

        for pet, id in zip(list_subjects, list_ids):
            # assuming that subject is already a anumoy array
            val_cort = np.mean(pet[mask_cort])
            cergy.append(val_cort / np.mean(pet[mask_ref_roi[0]]))
            whcer.append(val_cort / np.mean(pet[mask_ref_roi[1]]))
            cerbst.append(val_cort / np.mean(pet[mask_ref_roi[2]]))
            pons.append(val_cort / np.mean(pet[mask_ref_roi[3]]))

        return cergy, whcer, cerbst, pons

    def get_mask_regions(self):
        # step 10: Centiloid_std_VOI.zip ------------------- Standard Cortex and all four Reference Volumes-of-Interest
        #self.get_voi_gaain()
        mask_cort = self.vol_roi_cort == 1
        mask_cergy = self.vol_roi_cergy == 1
        mask_pons = self.vol_roi_pons == 1
        mask_whcer = self.vol_roi_whcer == 1
        mask_cerbst = self.vol_roi_cerbst == 1
        mask_ref_roi = [mask_cergy, mask_whcer, mask_cerbst, mask_pons]
        return mask_cort, mask_ref_roi

    def get_voi_gaain(self, path):
        # todo: hard-coded path. EIther save the .nii in the repor or dowsnload it from GAAIN
        #self.vol_roi_cort = np.array(nib.load(os.path.join(os.path.dirname(niftypad.cl_conversion.__file__), 'voi_ctx_2mm.nii')).dataobj)
        self.vol_roi_cort = np.array(nib.load(os.path.join(path, 'voi_ctx_2mm.nii')).dataobj)
        self.vol_roi_cergy = np.array(nib.load(os.path.join(path, 'voi_CerebGry_2mm.nii')).dataobj)
        self.vol_roi_pons = np.array(nib.load(os.path.join(path, 'voi_Pons_2mm.nii')).dataobj)
        self.vol_roi_whcer = np.array(nib.load(os.path.join(path, 'voi_WhlCbl_2mm.nii')).dataobj)
        self.vol_roi_cerbst = np.array(nib.load(os.path.join(path, 'voi_WhlCblBrnStm_2mm.nii')).dataobj)

    def calculate_mean_suvr(self, region):

        ad_select = self.df_subjects_gaain['group'].isin(['AD'])
        ad_mean = np.mean(self.df_subjects_gaain[region][ad_select].values)
        yc_select = self.df_subjects_gaain['group'].isin(['YC'])
        yc_mean = np.mean(self.df_subjects_gaain[region][yc_select].values)
        return ad_mean, yc_mean

    def linear_correlation(self, x, y):

        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

        # Plotting linear regression
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.gca()
        plt.scatter(x, y,  label='Centiloids')
        plt.plot(x, intercept + slope*x, 'r', label='fitted line')
        # ax.legend(loc=3, bbox_to_anchor=(0, 1.05), ncol=7, fancybox=True, shadow=True, prop={'size': 6})
        plt.grid(axis='y')
        str_add = 'y = ' + str(slope)+'x + ' + str(intercept)
        str_add = 'y = %.3fx '% slope + '+ %.3f' % intercept
        plt.text(0.6, 0.17, str_add)
        str_add_r = '$R^{2}$ =  %.3f' % r_value**2
        plt.text(0.6, 0.1, str_add_r)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)  # , prop={'size': 7})
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.xlim([-0.1, 100.1])
        ax.ylim([100.1, -0.1])
        ax.ylabel('Calculated Centiloids', fontsize=12, weight='bold')  # , **csfont)
        ax.xlabel('Reference Centiloids', fontsize=12, weight='bold')  # , **csfont)
        return slope, intercept, r_value


def run(args):
    # 1º Express that it can accurately express the level-1 PiB data on the Centiloid scale: download YC and AD subjects,
    # do normalization.., calculate CL and  show correlation between
    # their downloaded/recalculated PiB Centiloid values vs. the PiB Centiloid values reported here.
    FullCalibrationProcedure(args)




if __name__ == '__main__':
    # Poner con argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, default='standard', help='standard for Standard Pib Method '
                                                                     'for non-standard for other tracers method')
    parser.add_argument('--tracer_orig', type=str, default='pib', help='pib or f18')
    parser.add_argument('--tracer', type=str, default='Florbetaben', help='if non-standard method: possibility of tracer A, B or C')
    parser.add_argument('--folder_save', type=str, default='',  help='folder path')
    #parser.add_argument('--features_method', nargs="+", type=int, default=['tree-based'], help='number of features selected from Feature selection')

    args = parser.parse_args()
    #v.lower() in ('yes', 'true', 't', 'y', '1'):
    run(args)