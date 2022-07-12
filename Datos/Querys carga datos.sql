/****** tabla con labels de los datos  ******/
SELECT TOP (1000) 
	   [Item_Key]
      ,[Marca Kupfer]
FROM [dbo].[FEEDBACK_MP_Kupfer];

/******** Tabla con datos de productos = item *********/
SELECT [Tender_id]
      ,[Item_Key]
      ,[Rubro1]
      ,[Rubro2]
      ,[Rubro3]
      ,[Nombre linea Adquisicion]
      ,[Descripcion linea Adquisicion]
      ,[Kupfer]
FROM [dbo].[Dataset_MP_Kupfer];

/******** Tabla con datos de licitaciones = tender *********/
SELECT TOP (1000) 
	   [ocid]
      ,[Periodo]
      ,[Iid]
      ,[Lic_date]
      ,[Tag]
      ,[Lic_Name]
      ,[Parties_id]
      ,[Id_id]
      ,[Id_legalname]
      ,[Id_scheme]
      ,[StreetAddress]
      ,[Region]
      ,[Country]
      ,[Contactname]
      ,[Email]
      ,[Telephone]
      ,[Roles]
      ,[Tender_id]
      ,[Tender_title]
      ,[Tender_desc]
      ,[Tender_status]
      ,[Tender_value_am]
      ,[Tender_value_cur]
      ,[Tender_proc_name]
      ,[Tender_proc_name_id]
      ,[Tender_proc_id]
      ,[Tender_procmet]
      ,[Tender_procmetdet]
      ,[Startdate]
      ,[Enddate]
      ,[Duration]
      ,[Lic_Url]
  FROM [dbo].[LICITACIONES_APINUEVA_TENDER]


/***** Query que arma el dataset junta labels y detalle producto  ******/
SELECT 
	a.Tender_ID, a.Item_Key, a.Rubros1 AS Rubro1, a.Rubros2 AS Rubro2, a.Rubros3 AS Rubro3, a.Nombre_Producto, a.Descripcion, 
    b.[Marca Kupfer] AS Kupfer
FROM dbo.LICITACIONES_APIVIEJA_ITEMS AS a INNER JOIN
		dbo.FEEDBACK_MP_Kupfer AS b ON a.Item_Key = b.Item_Key;