'''
A module for data processing between from various formats
'''
import h5py
import labrad

from os.path import exists, join
import numpy as np
from traceback import format_exc

'''
nSOTColumnSpec allows generic nSOT data of particular types, corresponding to a specific filename
to be read in and unwrapped automatically.

Format is
{
"Name":[trace/retrace_index, reshape_order,
(column_axis_index, column_axis_values, column_label),
(row_axis_index, row_axis_values, row_labels),
(zvar_axis, index, zvar_axis_values, zvar_label), # Optional
(dependent_1, ..., dependent_N), (dependent_1_label, ..., dependent_N_label)]}

reshape_type is the "order" parameter to pass to reshape that determines how the elements are read
out. Options are "C", "F", and "A" see the numpy.reshape documentation. Generally "F" means the slow
axis is the first index and the fast index is the second.
"C" means the fast axis is the second axis.

trace/retrace_index should be negative if there is no such index

The dependent variables (and labels) can be ("*", ix) where all data columns starting with ix onwards
are assumed to be unspecified independent Variables. In which case the labels parameter (though it
should be present) is ignored and labels will be automatically generated.
'''
nSOTColumnSpec = {
# "nSOT vs. Bias Voltage and Field", ['Trace Index', 'B Field Index','Bias Voltage Index','B Field','Bias Voltage'],['DC SSAA Output','Noise']
"nSOT vs. Bias Voltage and Field":(0, "F", (1,3,"B Field (T)"), (2,4,"SQUID Bias (V)"), (5,6), ("Feedback (V)", "Noise")),
# "nSOT Scan Data " + self.fileName, ['Retrace Index','X Pos. Index','Y Pos. Index','X Pos. Voltage', 'Y Pos. Voltage'],in_name_list
"nSOT Scan Data unnamed":(0, "C", (1,3,"X Voltage"),(2,4,"Y Voltage"),('*',5),('*')),
# 'FourTerminal + self.Device_Name, ['Gate Voltage index','Gate Voltage'],["Voltage", "Current", "Resistance", "Conductance"]
"FourTerminal Device Name":(-1, "C", (0,1,"Gate Voltage"),(2,3,4,5), ("Voltage", "Current", "Resistance", "Conductance")),
# 'FourTerminal MagneticField ' + self.Device_Name, ['Magnetic Field index', 'Gate Voltage index', 'Magnetic Field', 'Gate Voltage'],["Voltage", "Current", "Resistance", "Conductance"]
"FourTerminal MagneticField Device Name":(-1, "C", (1,3,"Gate Voltage"), (0,2,"B Field"),(4,5,6,7), ("Voltage", "Current", "Resistance", "Conductance")),
# 'FourTerminal MagneticField ' + self.Device_Name, ['Magnetic Field index', 'Magnetic Field'],["Voltage", "Current", "Resistance", "Conductance"]
"1D Magnetic Field Device Name":(-1, "C", (0,1,"B Field"),(2,3,4,5), ("Voltage", "Current", "Resistance", "Conductance")),
# "Four Terminal Landau Voltage Biased", ['Gate Voltage index', 'Magnetic Field index',"Gate Voltage", "Magnetic Field"], ["Voltage Lock-In", "Current Lock-In"]
"Four Terminal Voltage Biased":(-1, "C", (0,2,"Gate Voltage"), (1,3,"B Field"),(4,5), ("Voltage", "Current")),
# "Dual Gate Voltage Biased Transport", ["p0 Index", "n0 Index","p0", "n0"], ["Vt", "Vb", "Voltage Lock-In", "Current Lock-In"]
"Dual Gate Voltage Biased Transport":(-1, "F", (0,2,"p0"), (1,3,"n0"), (4,5,6,7), ("Vt", "Vb", "Voltage", "Current")),
# 'n0p0 linecut vs DC bias sweep',['Trace_retrace','DC bias index','Gate index','DC bias value', 'n0 Value', 'p0 value','Bottom gate value', 'Top gate value'],['Idc','IacX','IacY','VacX','VacY'])
'n0p0 linecut vs DC bias sweep':(0,"F",(2,4,"n0"),(1,3,"Vbias"),(5,6,7,8,9,10,11,12),("p0","Vb","Vt", 'Idc','IacX','IacY','VacX','VacY')),
# '2D SQUID Transport Line',['Trace_retrace','Bottom gate index','Top gate index','Bottom gate value','Top gate value'],['SQUID_x','SQUID_y','I_x','V_x']
'2D SQUID Transport Line':(0, "C", (1,3,"Vb"), (2,4,"Vt"), (5,6,7,8), ('SQUID_x','SQUID_y','I_x','V_x')),
# dv.new('2D SQUID n0p0',['Trace_retrace','n0 index','p0 index', 'n0', 'p0','Bottom gate value','Top gate value'],['SQUID_x','SQUID_y','I_x','V_x'])
'2D SQUID n0p0':(0,"C",(1,3,'n0'),(2,4,'p0'), (5,6,7,8,9,10), ('Vb','Vt','SQUID_x','SQUID_y','I_x','V_x')),
# 'SQUID vs 2D Vector Magnet',['theta index','Bz index', 'theta', 'Bz'],['Bx','By','SQUID DC']
'SQUID vs 2D Vector Magnet':(-1, "C", (1,3,"Bz"), (0,2,"theta"),(4,5,6), ('Bx','By','SQUID DC')),
# '1D Displacement field vs DC bias sweep',['Trace_retrace','DC bias index','Bottom gate index','Top gate index','DC bias value','Bottom gate value', 'Top gate value'],['SQUID 1wx','Idc','Vac','Iac']
'1D Displacement field vs DC bias sweep':(0, "C", (1,4,"Vbias"), (3,6,"Vt"), (5,7,8,9,10), ('Vb','SQUID_x','I_dc','I_ac','V_ac')),
# '2D SQUID n0p0 vs bias field',['Trace_retrace','n0 index','p0 index','Field index', 'n0', 'p0','Bottom gate value','Top gate value', 'By', 'Bz', 'Bias'],['SQUID_x','SQUID_y','I_x','V_x']
'2D SQUID n0p0 vs bias field':(0, "C", (1,4,'n0'), (2,5,'p0'), (3,8,'By'), (6,7,9,10,11,12,13,14), ('Vb', 'Vt', 'Bz', 'Bias', 'SQUID_x', 'SQUID_y', 'I_x', 'V_x')),
# "2D SQUID Cap n0p0 - Sweep no RT", ("n0 index", "p0 index", 'n0', 'p0', cfg[measurement]['v1'], cfg[measurement]['v2']),('SQUID_x','SQUID_y','Cs', 'Ds', 'X', 'Y')
"2D SQUID Cap n0p0 sweep no RT":(-1,"C",(0,2,'n0'), (1,3,'p0'), (4,5,6,7,8,9,10,11), ('Vs', 'Vb', 'SQUID_x', 'SQUID_y', 'Cs', 'Ds', 'X', 'Y')),
# "2D SQUID Cap n0p0 linecut field", ("n0 index", "B index", 'n0', 'p0', cfg['magnet'], cfg[measurement]['v1'], cfg[measurement]['v2']),('SQUID_x','SQUID_y','Cs', 'Ds', 'X', 'Y'))
"2D SQUID Cap n0p0 linecut field":(-1,"C",(0,2,'n0'), (1,4,'B'), (3,5,6,7,8,9,10,11,12), ('p0','Vs', 'Vb', 'SQUID_x', 'SQUID_y', 'Cs', 'Ds', 'X', 'Y')),
# "2D SQUID Cap n0p0 DC no RT", ("n0 index", "p0 index", 'n0', 'p0', cfg[measurement]['v1'], cfg[measurement]['v2']),('SQUID_x','SQUID_y','SQUID_DC','Cs', 'Ds', 'X', 'Y')
"2D SQUID Cap n0p0 DC no RT":(-1,"C",(0,2,'n0'), (1,3,'p0'), (4,5,6,7,8,9,10,11,12), ('Vt', 'Vb', 'SQUID_x', 'SQUID_y', 'SQUID_DC', 'Cs', 'Ds', 'X', 'Y')),
# "2D SQUID Transport n0p0", ("n0 index", "p0 index", 'n0', 'p0', cfg[measurement]['v1'], cfg[measurement]['v2']),('SQUID_x','SQUID_y','Ix', 'Iy', 'V1x', 'V1y', 'V2x', 'V2y'))
"2D SQUID Transport n0p0":(-1,"C",(0,2,'n0'), (1,3,'p0'), (4,5,6,7,8,9,10,11,12,13), ('Vt', 'Vb', 'SQUID_x', 'SQUID_y', 'Ix', 'Iy', 'V1x', 'V1y', 'V2x', 'V2y')),
# "2D SQUID Transport n0p0 Field", ("n0 index", "B index", 'n0', 'p0', cfg['magnet'], cfg[measurement]['v1'], cfg[measurement]['v2']),('SQUID_x','SQUID_y','Ix', 'Iy', 'V1x', 'V1y', 'V2x', 'V2y'
"2D SQUID Transport n0p0 Field":(-1,"C",(0,2,'n0'), (1,4,'B'), (3,5,6,7,8,9,10,11,12,13, 14), ('p0', 'Vt', 'Vb', 'SQUID_x', 'SQUID_y', 'Ix', 'Iy', 'V1x', 'V1y', 'V2x', 'V2y')),
# 'SQUID video vs n0',['n0 index','n0', 'p0','Vb','Vt','x index','y index','x coordinate', 'y coordinate'],['SQUID 1wx', 'SQUID 1wy']
"SQUID video vs n0":(-1,"F", (5,7,'X'), (6,8,'Y'), (0,1,'n0'), (2,3,4,9,10), ('p0', 'Vb', 'Vt', 'SQUID_x', 'SQUID_y')),
}

def get_dv_data(identifier, remote=None, subfolder=None, params=False, retfilename=False, oldsystem=False):
    '''
    A function to retrieve data from the datavault using a nanosquid identifier and return is as numpy arrays

    Args:
        identifier (str): The specific
        remote (str): If not None will access data from a vault on another computer. This parameter
            is the remote name for the labrad.connect function
        subfolder : If not None access a subfolder within the vault. Works like an argument of the
            datavault.cd function, i.e. takes a String or list of strings forming a path to the folder.
        params (bool) : If True will return any parameters from the data vault file.
        retfilename : If True will return the name of the datavault file along with the data
        oldsystem : Identifier not using nanoSQUID Identifiers
    '''
    if remote is None:
        cxn = labrad.connect()
    else:
        cxn = labrad.connect(remote, password='pass')

    dv = cxn.data_vault
    if subfolder is not None:
        dv.cd(subfolder)

    drs, fls = dv.dir()
    if oldsystem:
        filename = [x for x in fls if identifier in x] # For things from other datavaults, not using nanoSQUID identifiers
    else:
        filename = [x for x in fls if identifier+" " in x] # the space prevents finding multiples of ten, for example iden-1 and iden-10

    if len(filename) == 0:
        raise IOError("Identifier " + identifier + " not found on this data vault.")
    elif len(filename) > 1:
        print("Warning files with duplicate identifiers detected, only the first one was retreived")
        print(filename)
    datafile = filename[0]


    # Java can't handel large datasets (what a wimp)
    # Cant do : data = np.array(dv.get())
    # Instead we load directly
    reg = cxn.registry
    reg.cd(['', 'Servers', 'Data Vault', 'Repository'])
    vault = reg.get('__default__')
    subfile = dv.cd()
    for x in subfile:
        if x != '':
            vault = join(vault, x +".dir")
    data = datavault2numpy(join(vault,datafile))

    if params:
        dv.open(datafile)
        plist = dv.get_parameters()
        parameters = dict()
        if plist is not None:
            for p in plist:
                if isfloat(p[1]):
                    parameters[p[0]] = float(p[1])
                else:
                    parameters[p[0]] = p[1]
        del dv # Somehow querying parameters keeps the files open?

    if retfilename and params:
        return data, parameters, datafile
    elif retfilename:
        return data, datafile
    elif params:
        return data, parameters
    else:
        return data

def retrievefromvault(vaultdir, filename):
    '''
    A generic tool to retrieve files from a LabRAD datavault

    Args:
        vaultdir (str) : The subdirectory of the vault ot find the files in (neglecting the .dir extension)
        filename (str) : The name of the file, neglecting the leading numbers or file extenstion, for
            example "data1" for the file "00001 - data1.hdf" if there are files with the same name but
            different numbers it will always retreive the first instance.
        host (str) : The host for the labrad connection, localhost by default.
        password (str) : The password for the labrad connection, localhost password by default.
    '''
    dv = labrad.connect('localhost', password='pass').data_vault
    for dir in vaultdir.split('\\'):
        dv.cd(dir)
    rt, fls = dv.dir()
    for fl in fls:
        if fl.split(' - ',1)[1] == filename:
            datafile = fl
            break
    dv.open(datafile)
    return np.array(dv.get())
#

def reshape_columns(data, shape=None, retrace=False):
    '''
    Simple function to take a data array and reshapes all columns of the array into new 2D arrays of size (rows, cols).
    If shape is None assumes the first two colums are the row index and column index respectively or if retrace is true
    then the first column is the trace/retrace index (with 0 for trace, 1 for retrace).

    Args:
        data: The data to reshape
        shape: The resulting arrays will have shape=(rows, cols). If None the shape will be determined as described above
        retrace: If the first column is the trace retract index it will return two sets based on the trace/retract index

    Returns:
        A list of arrays for each column of data of shape (rows, cols)
    '''
    if shape is None:
        # Subtract off minimum to deal with index starting from 1 which happens in some older scripts
        if retrace:
            rows = int(np.max(data[1])) - int(np.min(data[1])) + 1
            cols = int(np.max(data[2])) - int(np.min(data[2])) + 1
        else:
            rows = int(np.max(data[0])) - int(np.min(data[0])) + 1
            cols = int(np.max(data[1])) - int(np.min(data[1])) + 1
        shape = (rows, cols)

    if retrace:
        trace = data[data[:, 0] == 0, :]
        retrace = data[data[:, 0] == 1, :]
        dshp = trace.shape
        tr = []
        for j in range(dshp[1]):
            tr.append(np.reshape(trace[:, j], shape))
        rt = []
        for j in range(dshp[1]):
            rt.append(np.reshape(retrace[:, j], shape))
        return tr, rt
    else:
        dshp = data.shape
        ret = []
        for j in range(dshp[1]):
            ret.append(np.reshape(data[:,j], shape))
        return ret

def datavault2numpy(filename):
    '''
    A tool to convert a datavault hdf5 file into python numpy

    Args:
        filename (str) : The path to the datavault file, if .hdf5 extention is not included it will
            add it before attempting to load.
    '''
    ext_test = filename.split('.')
    if len(ext_test) > 1 and ext_test[len(ext_test)-1] == 'hdf5':
        pass
    else:
        filename = filename + '.hdf5'
    if not exists(filename):
        raise IOError("File " + str(filename) + " not found.")
    try:
        f = h5py.File(filename)
    except OSError:
        print("--------------------------------------------------------------")
        print("Could not open file", filename, "it may be locked for editing.")
        print("--------------------------------------------------------------")
        return None
    dv = f['DataVault']
    d = dv[...].tolist()
    return np.array(d)
#

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
    except TypeError:
        return False

def get_reshaped_nSOT_data(iden, quickload=None, overwrite=False, remote=None, subfolder=None, params=False, offbyone=False):
    '''
    Gets a data set of a known nSOT measurement type and unwraps it from columns into a useful
    dataset based on a known format. Assumes that it has a column that is the index for the fast
    and slow axes along with their values.

    Args:
        iden (str) : The Unique identifier generated by datavault for an nSOT system.
        quickload (str or None): Takes a directory or searches a directory for a .npz file with name iden
            and will load from that or save to it, if it doesn't exists.
        overwrite (bool): If True will reprocess and overwrite the quickload file.
        remote (str): If not None will access data from a vault on another computer. This parameter
            is the remote name for the labrad.connect function
        subfolder : If not None access a subfolder within the vault. Works like an argument of the
            datavault.cd function, i.e. takes a String or list of strings forming a path to the folder.
        params (bool) : If True will return any parameters from the data vault file.
        offbyone (bool) : Correct for the fact that some heathens 1 index their data.

    Returns in the format:
    row_values, colum_values, dependent_variables_trace, dependent_variables_retrace, labels
    Where dependent variables trace and retrace are in the order of the data vault and labels contains:
    (row_label, column_label, dependent_1_label, ..., dependent_N_label). If there is not distriction
    between trace and retrace then dependent_variables_trace and dependent_variables_retrace will be the same.

    '''
    if (quickload is not None) and not overwrite:
        if exists(quickload):
            if exists(join(quickload,iden + ".npz")):
                data = np.load(join(quickload,iden + ".npz"), allow_pickle=True)
                # rvalues, cvalues, dependent, dependent_retrace, labels, params
                if 'zvalues' in data:
                    if params:
                        dvparams = dict(data['params'].item())
                        return data['rvalues'], data['cvalues'], data['zvalues'], data['dependent'], data['dependent_retrace'], data['labels'], dvparams
                    else:
                        return data['rvalues'], data['cvalues'], data['zvalues'], data['dependent'], data['dependent_retrace'], data['labels']
                else:
                    if params:
                        dvparams = dict(data['params'].item())
                        return data['rvalues'], data['cvalues'], data['dependent'], data['dependent_retrace'], data['labels'], dvparams
                    else:
                        return data['rvalues'], data['cvalues'], data['dependent'], data['dependent_retrace'], data['labels']
        else:
            print("Quickload folder", quickload, "does not exist. Loading normally.")
    if params:
        d, dvparams, fname = get_dv_data(iden, remote=remote, subfolder=subfolder, retfilename=True, params=True)
    else:
        d, fname = get_dv_data(iden, remote=remote, subfolder=subfolder, retfilename=True, params=False)
        dvparams = None
    sweeptype = fname.split(' - ')[2]
    if sweeptype in nSOTColumnSpec:
        if len(nSOTColumnSpec[sweeptype]) > 6:
            rvalues, cvalues, zvalues, dependent, dependent_retrace, labels = reshape_from_spec_3d(d, nSOTColumnSpec[sweeptype], offbyone=offbyone, iden=iden)
            if (quickload is not None) and exists(quickload):
                np.savez(join(quickload,iden + ".npz"), rvalues=rvalues, cvalues=cvalues, zvalues=zvalues, dependent=dependent, dependent_retrace=dependent_retrace, labels=labels, params=dvparams)
            if params:
                return rvalues, cvalues, zvalues, dependent, dependent_retrace, labels, dvparams
            else:
                return rvalues, cvalues, zvalues, dependent, dependent_retrace, labels
        else:
            rvalues, cvalues, dependent, dependent_retrace, labels = reshape_from_spec(d, nSOTColumnSpec[sweeptype], offbyone=offbyone, iden=iden)
            if (quickload is not None) and exists(quickload):
                np.savez(join(quickload,iden + ".npz"), rvalues=rvalues, cvalues=cvalues, dependent=dependent, dependent_retrace=dependent_retrace, labels=labels, params=dvparams)
            if params:
                return rvalues, cvalues, dependent, dependent_retrace, labels, dvparams
            else:
                return rvalues, cvalues, dependent, dependent_retrace, labels
    else:
        raise ValueError("Unique Identifier does not correspond to a known 2D data type in nSOTColumnSpec")
#

def reshape_from_spec(d, spec, params=None, offbyone=False, iden=None):
    '''
    Takes an arbitrary set of data and reshapes it according to the spec, following normal convention

    Args:
        d (numpy array) : Data (as loaded straight from datavault) to reshape
        spec (tuple): The specification for the reshape (follows normal convention)
        params (tuple) : The datavult parameters for the data, to return as normal
        offbyone (bool) : Correct for the fact that some heathens 1 index their data.
            (Deprecated but left in for backwards compatibility)

    Returns in the format:
    row_values, colum_values, dependent_variables_trace, dependent_variables_retrace, labels
    Where dependent variables trace and retrace are in the order of the data vault and labels contains:
    (row_label, column_label, dependent_1_label, ..., dependent_N_label). If there is not distriction
    between trace and retrace then dependent_variables_trace and dependent_variables_retrace will be the same.

    '''
    trix, order, cvars, rvars, dvars, dvars_labels = spec

    try:
        if trix >= 0:
            trace = d[d[:,trix]==0,:]
            retrace = d[d[:,trix]==1,:]
        else:
            trace = d
            retrace = d

        if offbyone:
            print("offbyone option deprecated, should no longer be necessary")
            rows = int(np.max(trace[:,rvars[0]]))
            cols = int(np.max(trace[:,cvars[0]]))
        else:
            # Subtract off minimum to deal with index starting from 1 which happens in some older scripts
            rows = int(np.max(trace[:,rvars[0]])) - int(np.min(trace[:,rvars[0]])) + 1
            cols = int(np.max(trace[:,cvars[0]])) - int(np.min(trace[:,cvars[0]])) + 1

        #While data is still streaming in rows, cols may not match the actual number of rows
        nd = len(trace[:,rvars[1]])
        if rows*cols > nd:
            print(iden, "is not complete, loading partially.")
            print(nd, rows, cols)
            rows = nd//cols
        elif rows*cols < nd:
            print(iden, "warning row numbers off by one, attempting to correct.")
            rows += 1

        rvalues = np.reshape(trace[:,rvars[1]],(rows, cols), order=order)
        cvalues = np.reshape(trace[:,cvars[1]],(rows, cols), order=order)
    except ValueError:
        print("Error reshaping the data array, check that the specification is correct.")
        print(format_exc())
        return
    except IndexError:
        print("Error reshaping the data array, check that the specification is correct.")
        print(format_exc())
        print("data shape", d.shape)
        return

    dependent = []
    dependent_retrace = []
    if dvars[0] == "*":
        l, dcols = trace.shape
        dvars_labels = []
        for ix in range(dvars[1], dcols):
            tr = np.reshape(trace[:,ix],(rows, cols), order=order)
            dependent.append(np.array(tr))
            rt = np.reshape(retrace[:,ix],(rows, cols), order=order)
            dependent_retrace.append(np.array(rt))
            dvars_labels.append("Column "+str(ix))
    else:
        for ix in dvars:
            tr = np.reshape(trace[:,ix],(rows, cols), order=order)
            dependent.append(np.array(tr))
            rt = np.reshape(retrace[:,ix],(rows, cols), order=order)
            dependent_retrace.append(np.array(rt))
    labels = (rvars[2], cvars[2], *dvars_labels)

    if params is not None:
        return rvalues, cvalues, dependent, dependent_retrace, labels, params
    else:
        return rvalues, cvalues, dependent, dependent_retrace, labels
#

def reshape_from_spec_3d(d, spec, params=None, offbyone=False, iden=None):
    '''
    Takes an arbitrary set of data and reshapes it according to the spec, following normal convention

    Args:
        d (numpy array) : Data (as loaded straight from datavault) to reshape
        spec (tuple): The specification for the reshape (follows normal convention)
        params (tuple) : The datavult parameters for the data, to return as normal
        offbyone (bool) : Correct for the fact that some heathens 1 index their data.
            (Deprecated but left in for backwards compatibility)

    Returns in the format:
    row_values, colum_values, dependent_variables_trace, dependent_variables_retrace, labels
    Where dependent variables trace and retrace are in the order of the data vault and labels contains:
    (row_label, column_label, dependent_1_label, ..., dependent_N_label). If there is not distriction
    between trace and retrace then dependent_variables_trace and dependent_variables_retrace will be the same.

    '''
    trix, order, cvars, rvars, zvars, dvars, dvars_labels = spec

    try:
        if trix >= 0:
            trace = d[d[:,trix]==0,:]
            retrace = d[d[:,trix]==1,:]
        else:
            trace = d
            retrace = d

        if offbyone:
            print("offbyone option deprecated, should no longer be necessary")
            rows = int(np.max(trace[:,rvars[0]]))
            cols = int(np.max(trace[:,cvars[0]]))
            znum = int(np.max(trace[:,zvars[0]]))
        else:
            rows = int(np.max(trace[:, rvars[0]])) - int(np.min(trace[:, rvars[0]])) + 1
            cols = int(np.max(trace[:, cvars[0]])) - int(np.min(trace[:, cvars[0]])) + 1
            znum = int(np.max(trace[:, zvars[0]])) - int(np.min(trace[:, zvars[0]])) + 1

        # While data is still streaming in rows, cols may not match the actual number of rows
        nd, cn = trace.shape
        if rows * cols * znum > nd:
            print(iden, "is not complete, loading partially.")
            print(nd, rows, cols, znum)
            rows = rows - 1
            ix = int(rows * cols * znum)
            trace = trace[:ix,:]
            retrace = retrace[:ix, :]
        elif rows * cols < nd:
            print(iden, "warning row numbers off by one, attempting to correct.")
            rows += 1

        rvalues = np.reshape(trace[:,rvars[1]],(znum, rows, cols), order=order)
        cvalues = np.reshape(trace[:,cvars[1]],(znum, rows, cols), order=order)
        zvalues = np.reshape(trace[:,zvars[1]],(znum, rows, cols), order=order)
    except ValueError:
        print("Error reshaping the data array, check that the specification is correct.")
        print(format_exc())
        return

    dependent = []
    dependent_retrace = []
    if dvars[0] == "*":
        l, dcols = trace.shape
        dvars_labels = []
        for ix in range(dvars[1], dcols):
            tr = np.reshape(trace[:,ix],(znum, rows, cols), order=order)
            dependent.append(np.array(tr))
            rt = np.reshape(retrace[:,ix],(znum, rows, cols), order=order)
            dependent_retrace.append(np.array(rt))
            dvars_labels.append("Column "+str(ix))
    else:
        for ix in dvars:
            tr = np.reshape(trace[:,ix],(znum, rows, cols), order=order)
            dependent.append(np.array(tr))
            rt = np.reshape(retrace[:,ix],(znum, rows, cols), order=order)
            dependent_retrace.append(np.array(rt))
    labels = (rvars[2], cvars[2], zvars[2], *dvars_labels)

    if params is not None:
        return rvalues, cvalues, zvalues, dependent, dependent_retrace, labels, params
    else:
        return rvalues, cvalues, zvalues, dependent, dependent_retrace, labels
